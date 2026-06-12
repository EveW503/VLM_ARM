"""
VLM Bridge 节点: 双阶段 VLM 推理 + 2D→3D 映射 + 目标位姿发布。

阶段一: Gemini 335 全局推理 → 粗定位 → /target_pre_grasp
阶段二: ee_camera 近距精定位 → /target_pose
"""
import json
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
import tf2_ros

from .camera_utils import (
    get_min_depth_in_region,
    pixel_to_camera_3d,
    transform_point,
    compute_lookat_quaternion,
)
from .prompts import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    DEFAULT_STAGE1_USER_INSTRUCTION,
    DEFAULT_STAGE2_USER_INSTRUCTION,
)
from .vlm_client import VlmClient


# ── 目标排序 ─────────────────────────────────────────

RIPENESS_ORDER = {"ripe": 0, "overripe": 1, "unripe": 2}
OBSTACLE_ORDER = {"none": 0, "leaf": 1, "stem": 2}


def rank_targets(targets):
    """
    对 VLM 返回的目标列表按优先级排序。

    排序 key: 成熟度(ripe优先) > 上方无障碍 > 置信度高 > 原始索引

    Args:
        targets: list[dict], 每个 dict 包含:
            ripeness (str), obstacle_above (str), confidence (float)

    Returns:
        list[dict]: 排序后的目标列表 (最优在前)
    """
    def _sort_key(item):
        idx, t = item
        ripeness_score = RIPENESS_ORDER.get(t.get("ripeness", "unripe"), 2)
        obstacle_score = OBSTACLE_ORDER.get(t.get("obstacle_above", "leaf"), 2)
        # 置信度取负值 → 高置信度排前面
        return (ripeness_score, obstacle_score, -t.get("confidence", 0.0), idx)

    indexed = list(enumerate(targets))
    indexed.sort(key=_sort_key)
    return [t for _, t in indexed]


class VlmBridge(Node):
    STAGE_IDLE = "idle"
    STAGE1_QUERY = "stage1_query"
    STAGE1_WAIT = "stage1_wait"
    STAGE2_QUERY = "stage2_query"

    def __init__(self):
        super().__init__("vlm_bridge")

        # --- 参数 ---
        self.declare_parameter("dry_run", False)
        self._dry_run = self.get_parameter("dry_run").value

        # --- VLM 客户端 ---
        self._vlm = VlmClient()

        # --- TF2 ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Gemini 335 相机订阅 ---
        self._gemini_rgb = None
        self._gemini_depth = None
        self._gemini_info = None

        self.create_subscription(
            Image, "/camera/gemini_335/image_raw", self._gemini_rgb_cb, 10
        )
        self.create_subscription(
            Image, "/camera/gemini_335/depth/image_raw", self._gemini_depth_cb, 10
        )
        self.create_subscription(
            CameraInfo, "/camera/gemini_335/depth/camera_info", self._gemini_info_cb, 10
        )

        # --- ee_camera 手眼相机订阅 ---
        self._ee_rgb = None
        self._ee_depth = None
        self._ee_info = None

        self.create_subscription(
            Image, "/so101/camera/end_effector_depth_camera/image_raw", self._ee_rgb_cb, 10
        )
        self.create_subscription(
            Image, "/so101/camera/end_effector_depth_camera/depth/image_raw", self._ee_depth_cb, 10
        )
        self.create_subscription(
            CameraInfo, "/so101/camera/end_effector_depth_camera/depth/camera_info", self._ee_info_cb, 10
        )

        # --- 任务指令 ---
        self._task_instruction = DEFAULT_STAGE1_USER_INSTRUCTION
        self.create_subscription(
            String, "/task_command", self._task_cmd_cb, 10
        )

        # --- 抓取状态反馈 ---
        self.create_subscription(
            String, "/grab_status", self._grab_status_cb, 10
        )

        # --- 输出 ---
        self._pre_grasp_pub = self.create_publisher(
            PoseStamped, "/target_pre_grasp", 10
        )
        self._target_pub = self.create_publisher(
            PoseStamped, "/target_pose", 10
        )
        # --- 观察位姿发布 (新增) ---
        self._observation_pub = self.create_publisher(
            PoseStamped, "/target_observation_pose", 10
        )
        # --- 目标元数据发布 (材质、遮挡信息等) ---
        self._metadata_pub = self.create_publisher(
            String, "/target_metadata", 10
        )
        # --- 任务状态发布 (task_complete) ---
        self._task_status_pub = self.create_publisher(
            String, "/task_status", 10
        )

        # --- 状态机 ---
        self._lock = threading.Lock()
        self._state = self.STAGE_IDLE
        self._vlm_in_progress = False
        # 从 worker 线程到主线程的待发布消息队列
        self._pending_publish = None  # (topic_type, Point, quat_tuple) or None
        self._pending_rough = None    # (topic_type, Point) or None — P_rough 单独发布
        self._state_timer = self.create_timer(0.1, self._state_machine_tick)

        # --- 多目标状态 ---
        self._targets_pool = []       # 当前轮次排序后的目标列表
        self._current_target_idx = 0  # 当前选中目标在 pool 中的索引
        self._target_fail_counts = {}  # {label: fail_count} 跟踪每个目标失败次数
        self._target_blacklist = set()  # 黑名单目标标签集合
        self._consecutive_empty_stage1 = 0  # 连续空 Stage1 计数

        self.get_logger().info("VLM Bridge 节点已启动")

    # ── 回调: Gemini 335 ──────────────────────────────

    def _gemini_rgb_cb(self, msg):
        self._gemini_rgb = msg

    def _gemini_depth_cb(self, msg):
        self._gemini_depth = msg

    def _gemini_info_cb(self, msg):
        if self._gemini_info is None:
            self.get_logger().info(
                f"Gemini335 内参: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}"
            )
        self._gemini_info = msg

    # ── 回调: ee_camera ────────────────────────────────

    def _ee_rgb_cb(self, msg):
        self._ee_rgb = msg

    def _ee_depth_cb(self, msg):
        self._ee_depth = msg

    def _ee_info_cb(self, msg):
        if self._ee_info is None:
            self.get_logger().info(
                f"ee_camera 内参: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}"
            )
        self._ee_info = msg

    # ── 回调: 指令 / 状态 ──────────────────────────────

    def _task_cmd_cb(self, msg):
        self._task_instruction = msg.data
        self.get_logger().info(f"收到任务指令: {msg.data}")
        with self._lock:
            if self._state == self.STAGE_IDLE:
                self.get_logger().info("触发阶段一推理")
                self._state = self.STAGE1_QUERY

    def _grab_status_cb(self, msg):
        with self._lock:
            if msg.data == "stage1_done" and self._state == self.STAGE1_WAIT:
                self.get_logger().info("收到 stage1_done, 触发阶段二推理")
                self._state = self.STAGE2_QUERY
            elif msg.data == "grasp_complete":
                self.get_logger().info("收到 grasp_complete, 重新触发阶段一")
                self._targets_pool = []
                self._current_target_idx = 0
                self._target_fail_counts.clear()
                self._state = self.STAGE1_QUERY

            elif msg.data == "error":
                self.get_logger().warning("收到抓取失败(error)，尝试下一个目标")
                self._handle_grab_error()

    def _handle_grab_error(self):
        """处理抓取失败: 递增失败计数，如果超阈值则加入黑名单。"""
        if self._current_target_idx < len(self._targets_pool):
            target = self._targets_pool[self._current_target_idx]
            label = target.get("label", "unknown")
            bbox = target.get("bbox", [0, 0, 0, 0])
            target_key = f"{label}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"

            count = self._target_fail_counts.get(target_key, 0) + 1
            self._target_fail_counts[target_key] = count
            self.get_logger().warning(
                f"目标 '{label}' 失败 {count}/3 次"
            )
            if count >= 3:
                self.get_logger().error(
                    f"目标 '{label}' 连续失败 3 次, 加入黑名单"
                )
                self._target_blacklist.add(target_key)

            # 重新 Stage1 = rank_targets 会再次排序，黑名单目标被 process_current_target 跳过
            self.get_logger().info("重新 Stage1 以选择下一个目标")
        self._state = self.STAGE1_QUERY
        self._vlm_in_progress = False
        self._pending_publish = None
        self._pending_rough = None

    # ── 状态机 ─────────────────────────────────────────

    def _state_machine_tick(self):
        with self._lock:
            # 处理从 worker 线程来的待发布消息 (线程安全)
            if self._pending_publish is not None:
                topic_type, point, quat = self._pending_publish
                self._pending_publish = None
                msg = PoseStamped()
                msg.header.frame_id = "base"
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.position = point
                msg.pose.orientation.x = quat[0]
                msg.pose.orientation.y = quat[1]
                msg.pose.orientation.z = quat[2]
                msg.pose.orientation.w = quat[3]
                if topic_type == "observation":
                    self._observation_pub.publish(msg)
                    self.get_logger().info("已发布 /target_observation_pose")
                elif topic_type == "target":
                    self._target_pub.publish(msg)
                    self.get_logger().info("已发布 /target_pose")

            if self._pending_rough is not None:
                topic_type, point = self._pending_rough
                self._pending_rough = None
                msg = PoseStamped()
                msg.header.frame_id = "base"
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.position = point
                msg.pose.orientation.w = 1.0
                if topic_type == "pre_grasp":
                    self._pre_grasp_pub.publish(msg)
                    self.get_logger().info("已发布 /target_pre_grasp (P_rough)")

            if self._vlm_in_progress:
                return

            if self._state == self.STAGE1_QUERY:
                if self._gemini_rgb is None:
                    self.get_logger().warning("等待 Gemini335 RGB 图像...")
                    return
                if self._gemini_depth is None:
                    self.get_logger().warning("等待 Gemini335 深度图...")
                    return
                if self._gemini_info is None:
                    self.get_logger().warning("等待 Gemini335 camera_info...")
                    return
                self._vlm_in_progress = True
                self._state = self.STAGE1_QUERY  # keep state; thread will transition
                threading.Thread(target=self._run_stage1).start()

            elif self._state == self.STAGE2_QUERY:
                if self._ee_rgb is None:
                    self.get_logger().warning("等待 ee_camera RGB 图像...")
                    return
                if self._ee_depth is None:
                    self.get_logger().warning("等待 ee_camera 深度图...")
                    return
                if self._ee_info is None:
                    self.get_logger().warning("等待 ee_camera camera_info...")
                    return
                self._vlm_in_progress = True
                threading.Thread(target=self._run_stage2).start()

    # ── 阶段一: 全局推理 ───────────────────────────────

    def _run_stage1(self):
        try:
            self.get_logger().info("阶段一: 调用 VLM (Gemini335)...")
            rgb = self._gemini_rgb
            depth = self._gemini_depth
            info = self._gemini_info

            img_b64 = self._vlm.encode_image(rgb)
            result = self._vlm.call_vlm(
                STAGE1_SYSTEM_PROMPT, self._task_instruction, img_b64,
                max_tokens=500,
            )
            if result is None:
                self.get_logger().error("阶段一 VLM 调用失败")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            # 解析多目标列表
            targets = result.get("targets", [])
            obstacles = result.get("obstacles", [])

            if not targets:
                self.get_logger().warning(
                    f"VLM 未发现可采摘目标 (targets=[]), obstacles={len(obstacles)} 个"
                )
                self._consecutive_empty_stage1 += 1
                if self._consecutive_empty_stage1 >= 2:
                    self.get_logger().info("连续 2 次无目标, 任务完成")
                    self._publish_task_complete()
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return
            self._consecutive_empty_stage1 = 0

            self.get_logger().info(
                f"Stage1 发现 {len(targets)} 个目标, {len(obstacles)} 个障碍物"
            )

            # 排序: 成熟度 > 无障碍 > 置信度
            sorted_targets = rank_targets(targets)
            self._targets_pool = sorted_targets
            self._current_target_idx = 0

            # 处理当前最优目标
            success = self._process_current_target(rgb, depth, info)
            if not success:
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False

        except Exception as exc:
            self.get_logger().error(f"阶段一异常: {exc}")
            with self._lock:
                self._state = self.STAGE_IDLE
                self._vlm_in_progress = False

    def _process_current_target(self, rgb, depth, info):
        """
        从当前 targets_pool 中选取当前索引的目标，执行 2D→3D 映射并发布。

        Returns:
            bool: True 如果目标处理成功并已发布
        """
        while self._current_target_idx < len(self._targets_pool):
            target = self._targets_pool[self._current_target_idx]
            bbox = target.get("bbox", [-1, -1, -1, -1])
            confidence = target.get("confidence", 0.0)
            label = target.get("label", "unknown")
            direction = target.get("optimal_approach_direction", "top")
            material = target.get("material", "soft")
            ripeness = target.get("ripeness", "unripe")
            obstacle_above = target.get("obstacle_above", "none")

            # Check blacklist
            target_key = f"{label}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            if target_key in self._target_blacklist:
                self.get_logger().info(f"目标 {label} 在黑名单中, 跳过")
                self._current_target_idx += 1
                continue

            if bbox[0] < 0:
                self.get_logger().warning(f"目标 {label} bbox 无效, 跳过")
                self._current_target_idx += 1
                continue

            # Qwen3-VL bbox 坐标转换 [0,1000] → 像素
            x1 = int(bbox[0] / 1000.0 * rgb.width)
            y1 = int(bbox[1] / 1000.0 * rgb.height)
            x2 = int(bbox[2] / 1000.0 * rgb.width)
            y2 = int(bbox[3] / 1000.0 * rgb.height)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.get_logger().info(
                f"选中目标 [{self._current_target_idx}/{len(self._targets_pool)}]: "
                f"{label} ripeness={ripeness} material={material} "
                f"obstacle={obstacle_above} dir={direction} conf={confidence:.2f}"
            )

            Z = get_min_depth_in_region(depth, x1, y1, x2, y2)
            if Z is None:
                self.get_logger().error(f"目标 {label} 深度查询失败")
                self._current_target_idx += 1
                continue

            X_cam, Y_cam, Z_cam = pixel_to_camera_3d(cx, cy, Z, info)
            pt = transform_point(
                self._tf_buffer,
                X_cam, Y_cam, Z_cam,
                "camera_depth_optical_frame", "base",
            )
            if pt is None:
                self.get_logger().error(f"目标 {label} TF 变换失败")
                self._current_target_idx += 1
                continue

            p_rough = pt
            self.get_logger().info(
                f"3D rough (base): ({p_rough[0]:.4f}, {p_rough[1]:.4f}, {p_rough[2]:.4f})"
            )

            # 方位偏移 → 观察点
            DIRECTION_OFFSETS = {
                "front":       (-0.06,  0.00,  0.10),
                "left":        ( 0.00,  0.08,  0.10),
                "right":       ( 0.00, -0.08,  0.10),
                "front_left":  (-0.05,  0.06,  0.10),
                "front_right": (-0.05, -0.06,  0.10),
                "top":         ( 0.00,  0.00,  0.15),
            }
            offset = DIRECTION_OFFSETS.get(direction, DIRECTION_OFFSETS["top"])

            obs_x = p_rough[0] + offset[0]
            obs_y = p_rough[1] + offset[1]
            obs_z = p_rough[2] + offset[2]

            if obs_x < 0.10:
                self.get_logger().warn(
                    f"观察点 X={obs_x:.3f} 进入基座死区, 强制钳制至 X=0.10"
                )
                obs_x = 0.10

            p_obs = (obs_x, obs_y, obs_z)
            q = compute_lookat_quaternion(p_obs, p_rough)
            if q is None:
                self.get_logger().error("LookAt 四元数计算失败, 降级为 identity")
                q = (0.0, 0.0, 0.0, 1.0)

            self.get_logger().info(
                f"观察位姿 (base): pos=({p_obs[0]:.4f}, {p_obs[1]:.4f}, {p_obs[2]:.4f})"
            )

            with self._lock:
                if not self._dry_run:
                    self._pending_publish = (
                        "observation",
                        Point(x=p_obs[0], y=p_obs[1], z=p_obs[2]),
                        q,
                    )
                    self._pending_rough = (
                        "pre_grasp",
                        Point(x=p_rough[0], y=p_rough[1], z=p_rough[2]),
                    )
                self._state = self.STAGE1_WAIT
                self._vlm_in_progress = False

            # 发布元数据
            if not self._dry_run:
                self._publish_metadata(material, obstacle_above, direction, ripeness, label)

            return True

        self.get_logger().warning("targets_pool 已耗尽, 需要重新 Stage1")
        return False

    def _publish_metadata(self, material, obstacle_above, direction, ripeness, label):
        """发布当前目标的语义元数据到 /target_metadata。"""
        meta = {
            "material": material,
            "obstacle_above": obstacle_above,
            "optimal_approach_direction": direction,
            "ripeness": ripeness,
            "label": label,
        }
        msg = String()
        msg.data = json.dumps(meta, ensure_ascii=False)
        self._metadata_pub.publish(msg)
        self.get_logger().info(f"已发布 /target_metadata: {msg.data}")

    def _publish_task_complete(self):
        """发布任务完成信号。"""
        msg = String()
        msg.data = "task_complete"
        self._task_status_pub.publish(msg)
        self.get_logger().info("已发布 /task_status: task_complete — 采摘任务结束")

    # ── 阶段二: 精确定位 ───────────────────────────────

    def _run_stage2(self):
        try:
            self.get_logger().info("阶段二: 调用 VLM (ee_camera)...")
            rgb = self._ee_rgb
            depth = self._ee_depth
            info = self._ee_info

            img_b64 = self._vlm.encode_image(rgb)
            result = self._vlm.call_vlm(
                STAGE2_SYSTEM_PROMPT, DEFAULT_STAGE2_USER_INSTRUCTION, img_b64
            )
            if result is None:
                self.get_logger().error("阶段二 VLM 调用失败")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            target = result.get("target", {})
            bbox = target.get("bbox", [-1, -1, -1, -1])
            confidence = target.get("confidence", 0.0)
            label = target.get("label", "unknown")

            if bbox[0] < 0:
                self.get_logger().warning("VLM 阶段二未发现目标")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            x1 = int(bbox[0] / 1000.0 * rgb.width)
            y1 = int(bbox[1] / 1000.0 * rgb.height)
            x2 = int(bbox[2] / 1000.0 * rgb.width)
            y2 = int(bbox[3] / 1000.0 * rgb.height)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.get_logger().info(
                f"阶段二结果: bbox_norm={bbox} → pixel=[{x1},{y1},{x2},{y2}] "
                f"center=[{cx},{cy}] label={label} conf={confidence:.2f}"
            )

            Z = get_min_depth_in_region(depth, x1, y1, x2, y2)
            if Z is None:
                self.get_logger().error("阶段二: 深度查询失败")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            X_cam, Y_cam, Z_cam = pixel_to_camera_3d(cx, cy, Z, info)
            pt = transform_point(
                self._tf_buffer,
                X_cam, Y_cam, Z_cam,
                "ee_camera_optical_link", "base",
            )
            if pt is None:
                self.get_logger().error("阶段二: TF 变换失败 (ee_camera_optical_link→base)")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            self.get_logger().info(
                f"阶段二 3D (base): ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
            )

            with self._lock:
                if not self._dry_run:
                    self._pending_publish = (
                        "target",
                        Point(x=pt[0], y=pt[1], z=pt[2]),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                self._pending_rough = None  # 清除阶段一的粗定位，防止残留
                self._state = self.STAGE_IDLE
                self._vlm_in_progress = False

        except Exception as exc:
            self.get_logger().error(f"阶段二异常: {exc}")
            with self._lock:
                self._state = self.STAGE_IDLE
                self._vlm_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = VlmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
