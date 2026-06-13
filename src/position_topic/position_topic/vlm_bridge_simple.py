"""
VLM Bridge 节点 (简化版): 仅阶段一 VLM 推理 + 2D→3D 映射 + P_rough 发布。

与完整版的区别:
- 去掉阶段二精确定位 (STAGE2_QUERY)
- 去掉观察位姿系统 (/target_observation_pose)
- 去掉 ee_camera 手眼相机订阅
- 阶段一完成后直接发布 P_rough, 不等待 stage1_done
"""
import json
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
import tf2_ros

from .camera_utils import (
    get_min_depth_in_region,
    get_depth_at_pixel,
    pixel_to_camera_3d,
    transform_point,
)
from .prompts import (
    STAGE1_SYSTEM_PROMPT,
    DEFAULT_STAGE1_USER_INSTRUCTION,
    GENERIC_STAGE1_SYSTEM_PROMPT,
    GENERIC_STAGE1_USER_INSTRUCTION,
)
from .vlm_client import VlmClient


# ── 目标排序 ─────────────────────────────────────────

RIPENESS_ORDER = {"ripe": 0, "overripe": 1, "unripe": 2}
OBSTACLE_ORDER = {"none": 0, "leaf": 1, "stem": 2}


def rank_targets(targets):
    """
    对 VLM 返回的目标列表按优先级排序。

    排序 key: 成熟度(ripe优先) > 上方无障碍 > 置信度高 > 原始索引
    """
    def _sort_key(item):
        idx, t = item
        ripeness_score = RIPENESS_ORDER.get(t.get("ripeness", "unripe"), 2)
        obstacle_score = OBSTACLE_ORDER.get(t.get("obstacle_above", "leaf"), 2)
        return (ripeness_score, obstacle_score, -t.get("confidence", 0.0), idx)

    indexed = list(enumerate(targets))
    indexed.sort(key=_sort_key)
    return [t for _, t in indexed]


class VlmBridgeSimple(Node):
    """简化版 VLM Bridge: 仅阶段一, 无观察位姿, 无阶段二。"""

    STAGE_IDLE = "idle"
    STAGE1_QUERY = "stage1_query"

    def __init__(self):
        super().__init__("vlm_bridge_simple")

        # --- 参数 ---
        self.declare_parameter("dry_run", False)
        self.declare_parameter("generic_mode", False)
        self._dry_run = self.get_parameter("dry_run").value
        self._generic_mode = self.get_parameter("generic_mode").value

        if self._generic_mode:
            self._stage1_system_prompt = GENERIC_STAGE1_SYSTEM_PROMPT
            self._stage1_user_instruction = GENERIC_STAGE1_USER_INSTRUCTION
            self.get_logger().info("通用抓取模式 (generic_mode=True)")
        else:
            self._stage1_system_prompt = STAGE1_SYSTEM_PROMPT
            self._stage1_user_instruction = DEFAULT_STAGE1_USER_INSTRUCTION

        # --- VLM 客户端 ---
        self._vlm = VlmClient()

        # --- TF2 ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Gemini 335 相机订阅 (仅全局相机) ---
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

        # --- 任务指令 ---
        self._task_instruction = self._stage1_user_instruction
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
        self._metadata_pub = self.create_publisher(
            String, "/target_metadata", 10
        )
        self._task_status_pub = self.create_publisher(
            String, "/task_status", 10
        )

        # --- 状态机 ---
        self._lock = threading.Lock()
        self._state = self.STAGE_IDLE
        self._vlm_in_progress = False
        self._vlm_generation = 0
        # 从 worker 线程到主线程的待发布 P_rough
        self._pending_rough = None  # Point or None
        self._state_timer = self.create_timer(0.1, self._state_machine_tick)

        # --- 多目标状态 ---
        self._targets_pool = []
        self._current_target_idx = 0
        self._target_fail_counts = {}
        self._target_blacklist = set()
        self._consecutive_empty_stage1 = 0

        self.get_logger().info("VLM Bridge Simple 节点已启动 (无阶段二/无观察位姿)")

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
            if msg.data == "grasp_complete":
                self.get_logger().info("收到 grasp_complete, 重新触发阶段一")
                self._targets_pool = []
                self._current_target_idx = 0
                self._target_fail_counts.clear()
                self._vlm_generation += 1
                self._state = self.STAGE1_QUERY

            elif msg.data == "error":
                self.get_logger().warning("收到抓取失败(error)，尝试下一个目标")
                self._vlm_generation += 1
                self._handle_grab_error()

    def _handle_grab_error(self):
        """处理抓取失败: 递增失败计数，超阈值加入黑名单，重新 Stage1。"""
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

            self.get_logger().info("重新 Stage1 以选择下一个目标")
        self._state = self.STAGE1_QUERY
        self._vlm_in_progress = False
        self._pending_rough = None

    # ── 状态机 ─────────────────────────────────────────

    def _state_machine_tick(self):
        with self._lock:
            # 处理从 worker 线程来的待发布 P_rough
            if self._pending_rough is not None:
                point = self._pending_rough
                self._pending_rough = None
                msg = PoseStamped()
                msg.header.frame_id = "world"
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.position = point
                msg.pose.orientation.w = 1.0
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
                self._vlm_generation += 1
                gen = self._vlm_generation
                self._vlm_in_progress = True
                threading.Thread(target=self._run_stage1, args=(gen,)).start()

    # ── 阶段一: 全局推理 ───────────────────────────────

    def _run_stage1(self, gen):
        try:
            self.get_logger().info("阶段一: 调用 VLM (Gemini335)...")
            rgb = self._gemini_rgb
            depth = self._gemini_depth
            info = self._gemini_info

            img_b64 = self._vlm.encode_image(rgb)
            result = self._vlm.call_vlm(
                self._stage1_system_prompt, self._task_instruction, img_b64,
                max_tokens=500,
            )
            if result is None:
                self.get_logger().error("阶段一 VLM 调用失败")
                with self._lock:
                    if self._vlm_generation != gen:
                        return
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
                    if self._vlm_generation != gen:
                        return
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return
            self._consecutive_empty_stage1 = 0

            self.get_logger().info(
                f"Stage1 发现 {len(targets)} 个目标, {len(obstacles)} 个障碍物"
            )

            # 排序
            sorted_targets = rank_targets(targets)
            self._targets_pool = sorted_targets
            self._current_target_idx = 0

            # 处理当前最优目标
            success = self._process_current_target(rgb, depth, info, gen)
            if not success:
                with self._lock:
                    if self._vlm_generation != gen:
                        return
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False

        except Exception as exc:
            self.get_logger().error(f"阶段一异常: {exc}")
            with self._lock:
                if self._vlm_generation != gen:
                    return
                self._state = self.STAGE_IDLE
                self._vlm_in_progress = False

    def _process_current_target(self, rgb, depth, info, gen):
        """
        从 targets_pool 选取当前索引目标，执行 2D→3D 映射并发布 P_rough。

        简化版: 不计算观察位姿，不发布 /target_observation_pose。
        发布 /target_pre_grasp + /target_metadata 后直接回到 IDLE。
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

            # 深度查询
            Z = get_depth_at_pixel(depth, cx, cy)
            if Z is None:
                self.get_logger().warn(f"目标 {label} 中心深度无效, 回退到区域最小深度")
                Z = get_min_depth_in_region(depth, x1, y1, x2, y2)
            if Z is None:
                self.get_logger().error(f"目标 {label} 深度查询失败")
                self._current_target_idx += 1
                continue
            self.get_logger().info(f"深度查询: Z={Z:.4f} (pixel=[{cx},{cy}])")

            X_cam, Y_cam, Z_cam = pixel_to_camera_3d(cx, cy, Z, info)
            pt = transform_point(
                self._tf_buffer,
                X_cam, Y_cam, Z_cam,
                "camera_depth_optical_frame", "world",
            )
            if pt is None:
                self.get_logger().error(f"目标 {label} TF 变换失败")
                self._current_target_idx += 1
                continue

            p_rough = pt
            self.get_logger().info(
                f"3D rough (world): ({p_rough[0]:.4f}, {p_rough[1]:.4f}, {p_rough[2]:.4f})"
            )

            if not all(np.isfinite(p_rough)):
                self.get_logger().error(
                    f"阶段一 3D rough 无效: ({p_rough[0]:.4f}, {p_rough[1]:.4f}, {p_rough[2]:.4f})"
                )
                self._current_target_idx += 1
                continue

            # 合理性检查
            if p_rough[2] < 0.02:
                self.get_logger().warn(
                    f"阶段一 Z_rough={p_rough[2]:.4f} 极低 (<0.02m base)，深度可能穿透物体。"
                )

            # 发布 P_rough + 元数据, 然后回到 IDLE
            with self._lock:
                if self._vlm_generation != gen:
                    return False
                if not self._dry_run:
                    self._pending_rough = Point(
                        x=p_rough[0], y=p_rough[1], z=p_rough[2]
                    )
                self._state = self.STAGE_IDLE
                self._vlm_in_progress = False

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


def main(args=None):
    rclpy.init(args=args)
    node = VlmBridgeSimple()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
