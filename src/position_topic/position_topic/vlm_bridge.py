"""
VLM Bridge 节点: 双阶段 VLM 推理 + 2D→3D 映射 + 目标位姿发布。

阶段一: Gemini 335 全局推理 → 粗定位 → /target_pre_grasp
阶段二: ee_camera 近距精定位 → /target_pose
"""
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
import tf2_ros

from .camera_utils import get_depth_at_pixel, pixel_to_camera_3d, transform_point
from .prompts import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    DEFAULT_STAGE1_USER_INSTRUCTION,
    DEFAULT_STAGE2_USER_INSTRUCTION,
)
from .vlm_client import VlmClient


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

        # --- 状态机 ---
        self._lock = threading.Lock()
        self._state = self.STAGE_IDLE
        self._vlm_in_progress = False
        # 从 worker 线程到主线程的待发布消息队列
        self._pending_publish = None  # (topic_type, Point) or None
        self._state_timer = self.create_timer(0.1, self._state_machine_tick)

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

    # ── 状态机 ─────────────────────────────────────────

    def _state_machine_tick(self):
        # 处理从 worker 线程来的待发布消息 (主线程安全)
        if self._pending_publish is not None:
            topic_type, point = self._pending_publish
            self._pending_publish = None
            msg = PoseStamped()
            msg.header.frame_id = "base"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position = point
            msg.pose.orientation.w = 1.0
            if topic_type == "pre_grasp":
                self._pre_grasp_pub.publish(msg)
                self.get_logger().info("已发布 /target_pre_grasp")
            elif topic_type == "target":
                self._target_pub.publish(msg)
                self.get_logger().info("已发布 /target_pose")

        with self._lock:
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
                STAGE1_SYSTEM_PROMPT, self._task_instruction, img_b64
            )
            if result is None:
                self.get_logger().error("阶段一 VLM 调用失败")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            target = result.get("target", {})
            px = target.get("pixel_x", -1)
            py = target.get("pixel_y", -1)
            confidence = target.get("confidence", 0.0)
            label = target.get("label", "unknown")

            if px < 0 or py < 0:
                self.get_logger().warning("VLM 未发现可采摘目标")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            self.get_logger().info(
                f"阶段一结果: ({px}, {py}) label={label} conf={confidence:.2f}"
            )

            Z = get_depth_at_pixel(depth, px, py)
            if Z is None:
                self.get_logger().error("阶段一: 深度查询失败")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            X_cam, Y_cam, Z_cam = pixel_to_camera_3d(px, py, Z, info)
            pt = transform_point(
                self._tf_buffer,
                X_cam, Y_cam, Z_cam,
                "camera_depth_optical_frame", "base",
            )
            if pt is None:
                self.get_logger().error("阶段一: TF 变换失败 (camera_depth_optical_frame→base)")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            self.get_logger().info(
                f"阶段一 3D (base): ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
            )

            if not self._dry_run:
                self._pending_publish = ("pre_grasp", Point(x=pt[0], y=pt[1], z=pt[2]))

            with self._lock:
                self._state = self.STAGE1_WAIT
                self._vlm_in_progress = False

        except Exception as exc:
            self.get_logger().error(f"阶段一异常: {exc}")
            with self._lock:
                self._state = self.STAGE_IDLE
                self._vlm_in_progress = False

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
            px = target.get("pixel_x", -1)
            py = target.get("pixel_y", -1)
            confidence = target.get("confidence", 0.0)
            label = target.get("label", "unknown")

            if px < 0 or py < 0:
                self.get_logger().warning("VLM 阶段二未发现目标")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            self.get_logger().info(
                f"阶段二结果: ({px}, {py}) label={label} conf={confidence:.2f}"
            )

            Z = get_depth_at_pixel(depth, px, py)
            if Z is None:
                self.get_logger().error("阶段二: 深度查询失败")
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return

            X_cam, Y_cam, Z_cam = pixel_to_camera_3d(px, py, Z, info)
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

            if not self._dry_run:
                self._pending_publish = ("target", Point(x=pt[0], y=pt[1], z=pt[2]))

            with self._lock:
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
