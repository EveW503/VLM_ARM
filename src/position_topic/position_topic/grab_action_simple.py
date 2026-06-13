"""
Grab Action 节点 (简化版): 抓取全流程状态机，无阶段二/无观察位姿。

与完整版的区别:
- 去掉 MOVE_TO_OBSERVE 状态 (观察位姿移动)
- 去掉 AWAIT_STAGE2 状态 (等待阶段二精定位)
- 去掉 /target_observation_pose 和 /target_pose 订阅
- SETUP 完成后直接进入 PRE_GRASP, 使用 P_rough 作为抓取目标
- 状态数从 11 减为 9

状态流: IDLE → SETUP → PRE_GRASP → [PUSH_SWEEP] → APPROACH
         → GRASP → LIFT_PLAN → TRANSPORT_PLAN → PLACE → IDLE
"""
import json
import math
import time
import traceback

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume, Constraints, MotionPlanRequest,
    OrientationConstraint, PlanningOptions, PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from linkattacher_msgs.srv import AttachLink, DetachLink
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — register geometry_msgs TF2 support

from .planning_scene_manager import PlanningSceneManager


class GrabActionSimple(Node):
    """简化版 Grab Action: 无观察位姿, 无阶段二, P_rough 即抓取目标。"""

    # 状态常量 (9 状态, 去掉 MOVE_TO_OBSERVE / AWAIT_STAGE2)
    IDLE = "idle"
    SETUP = "setup"
    PRE_GRASP = "pre_grasp"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT_PLAN = "lift_plan"
    TRANSPORT_PLAN = "transport_plan"
    PLACE = "place"
    PUSH_SWEEP = "push_sweep"
    FAILED = "failed"

    def __init__(self):
        super().__init__("grab_action_simple")

        # --- 参数 ---
        self.declare_parameter("placement_position", [0.2, -0.15, 0.25])
        self.declare_parameter("pre_grasp_z_offset", 0.08)
        self.declare_parameter("gripper_close_pos", -0.17)
        self.declare_parameter("gripper_open_pos", 0.5)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.3)
        self.declare_parameter("approach_mode", "auto")
        self.declare_parameter("target_object_shape", "BOX")
        self.declare_parameter("target_object_dims", [0.05, 0.05, 0.05])
        self.declare_parameter("jaw_clearance", 0.00)   # 夹爪几何补偿(P_rough已修正到物体中心后的小偏移)

        # --- Planning Scene Manager ---
        self._psm = PlanningSceneManager(self)

        # --- TF2 (查询实际末端姿态) ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- 输入订阅 (仅 P_rough + 元数据) ---
        self.create_subscription(
            PoseStamped, "/target_pre_grasp", self._pre_grasp_cb, 10
        )
        self.create_subscription(
            String, "/target_metadata", self._metadata_cb, 10
        )

        # --- 状态反馈 ---
        self._status_pub = self.create_publisher(String, "/grab_status", 10)

        # --- MoveGroup action client ---
        self._move_action = ActionClient(self, MoveGroup, "/move_action")
        if not self._move_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("无法连接 /move_action")

        # --- LinkAttacher service clients ---
        self._attach_cli = self.create_client(AttachLink, "/ATTACHLINK")
        self._detach_cli = self.create_client(DetachLink, "/DETACHLINK")

        # --- Gripper action client ---
        self._gripper_action = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

        # --- Cartesian path service + arm trajectory execution ---
        self._cartesian_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._arm_traj_action = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )

        # --- 状态机 ---
        self._state = self.IDLE
        self._pre_grasp_target = None   # P_rough, 即是碰撞注册位置也是抓取目标
        self._target_metadata = {}
        self._consecutive_failures = 0
        self._sweep_in_progress = False
        self._pre_grasp_orientation = None
        self._target_object_registered = False

        self.get_logger().info("Grab Action Simple 节点已启动 (无观察位姿/无阶段二)")

    # ── 回调 ──────────────────────────────────────────

    def _pre_grasp_cb(self, msg):
        """接收 P_rough, 直接触发 SETUP (无需等待观察位姿)。"""
        if self._state != self.IDLE:
            return
        self.get_logger().info(
            f"收到 P_rough: ({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )
        self._pre_grasp_target = msg
        self._target_object_registered = False
        self._transition(self.SETUP)

    def _metadata_cb(self, msg):
        """接收当前目标的语义元数据。"""
        try:
            self._target_metadata = json.loads(msg.data)
            self.get_logger().info(
                f"收到元数据: material={self._target_metadata.get('material')}, "
                f"obstacle={self._target_metadata.get('obstacle_above')}, "
                f"direction={self._target_metadata.get('optimal_approach_direction')}"
            )
        except json.JSONDecodeError:
            self.get_logger().warning(f"无法解析 /target_metadata: {msg.data}")

    # ── 状态机驱动 ────────────────────────────────────

    def _transition(self, new_state):
        self._state = new_state
        try:
            if new_state == self.SETUP:
                self._do_setup()
            elif new_state == self.PRE_GRASP:
                self._plan_pre_grasp()
            elif new_state == self.PUSH_SWEEP:
                self._do_push_sweep()
            elif new_state == self.APPROACH:
                self._do_approach()
            elif new_state == self.LIFT_PLAN:
                self._plan_lift()
            elif new_state == self.TRANSPORT_PLAN:
                self._plan_transport()
            elif new_state == self.GRASP:
                self._do_grasp()
            elif new_state == self.PLACE:
                self._do_place()
            elif new_state == self.FAILED:
                self.get_logger().error("抓取流程失败")
                self._consecutive_failures += 1
                self._publish_status("error")
                if self._target_object_registered:
                    self._psm.remove_object("target")
                    self._target_object_registered = False
                self._state = self.IDLE
        except Exception as exc:
            self.get_logger().error(f"{new_state} 异常: {exc}")
            self.get_logger().error(traceback.format_exc())
            self._publish_status("error")
            if self._target_object_registered:
                self._psm.remove_object("target")
                self._target_object_registered = False
            self._consecutive_failures += 1
            self._state = self.IDLE

    def _publish_status(self, status):
        self._status_pub.publish(String(data=status))

    # ── SETUP ─────────────────────────────────────

    def _do_setup(self):
        try:
            self._do_setup_impl()
        except Exception as exc:
            self.get_logger().error(f"SETUP 异常: {exc}")
            self.get_logger().error(traceback.format_exc())
            self._transition(self.FAILED)

    def _do_setup_impl(self):
        self.get_logger().info("SETUP: 注册目标物体到 Planning Scene...")
        target = self._pre_grasp_target
        if target is None:
            self._transition(self.FAILED)
            return

        shape_map = {"BOX": SolidPrimitive.BOX, "SPHERE": SolidPrimitive.SPHERE,
                     "CYLINDER": SolidPrimitive.CYLINDER}
        shape_type = shape_map.get(
            self.get_parameter("target_object_shape").value, SolidPrimitive.BOX
        )
        dims = self.get_parameter("target_object_dims").value

        # VLM Z = 物体上表面 → Planning Scene 中心 = Z - 半高
        obj_pose = Pose()
        obj_pose.position = Point(
            x=target.pose.position.x,
            y=target.pose.position.y,
            z=target.pose.position.z - dims[2] / 2.0,
        )
        obj_pose.orientation = target.pose.orientation

        self._psm.add_object(
            "target", shape_type, dims,
            obj_pose, target.header.frame_id or "world"
        )
        self._target_object_registered = True
        self._consecutive_failures = 0
        self.get_logger().info(f"目标已注册 (中心 Z={obj_pose.position.z:.3f})")
        # SETUP 完成 → 直接进入 PRE_GRASP
        self._transition(self.PRE_GRASP)

    # ── 预抓取 (目标正上方) ─────────────────────

    def _plan_pre_grasp(self):
        """移动到 P_rough 正上方 (抬高到安全高度, 使手臂自然形成向下姿态)。"""
        self.get_logger().info("PRE_GRASP: 移动到目标正上方 (不约束姿态)...")

        if self._pre_grasp_target is None:
            self._transition(self.FAILED)
            return

        # 物体留在 Planning Scene 中, MoveIt 规划时会绕行
        # 等到 APPROACH Cartesian 下探前再移除
        if self._target_object_registered:
            self.get_logger().info("PRE_GRASP: 物体留在场景中, MoveIt 自行避让")

        z_offset = self.get_parameter("pre_grasp_z_offset").value

        # 钳制最低高度: PRE_GRASP 位置太低时手臂姿态侧偏, 必须抬高到安全 Z
        pre_grasp_z = self._pre_grasp_target.pose.position.z + z_offset
        safe_z_min = 0.20  # base 系下最低安全高度, 保证手臂自然向下
        if pre_grasp_z < safe_z_min:
            self.get_logger().warn(
                f"PRE_GRASP Z={pre_grasp_z:.3f} 低于安全最低 {safe_z_min:.3f}, 钳制"
            )
            pre_grasp_z = safe_z_min

        pre_grasp_pose = Pose()
        pre_grasp_pose.position = Point(
            x=self._pre_grasp_target.pose.position.x,
            y=self._pre_grasp_target.pose.position.y,
            z=pre_grasp_z,
        )
        # 强制末端垂直向下: Ry(-90°) → gripper X 指向世界 -Z (接近方向)
        sqrt2_2 = math.sqrt(2) / 2.0
        pre_grasp_pose.orientation = Quaternion(x=0.0, y=-sqrt2_2, z=0.0, w=sqrt2_2)

        self._pre_grasp_orientation = None  # 到达后从 TF 查询

        self._send_move_request(
            pre_grasp_pose, self._pre_grasp_target.header.frame_id or "world",
            success_callback=self._on_pre_grasp_done,
            fail_callback=lambda: self._transition(self.FAILED),
            use_orientation=True,
            pos_tolerance=0.03,
        )

    def _lookup_gripper_orientation(self):
        """查询 base→gripper 的实际 TF 姿态, 供 APPROACH 沿用实现纯平移下探。"""
        import rclpy.time
        try:
            t = self._tf_buffer.lookup_transform(
                "world", "gripper",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            q = Quaternion(
                x=t.transform.rotation.x,
                y=t.transform.rotation.y,
                z=t.transform.rotation.z,
                w=t.transform.rotation.w,
            )
            self.get_logger().info(
                f"查询到 gripper 实际姿态: "
                f"({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
            )
            return q
        except Exception as exc:
            self.get_logger().warn(f"TF 查询 gripper 姿态失败: {exc}, 降级为 identity")
            return Quaternion(w=1.0)

    def _on_pre_grasp_done(self):
        """PRE_GRASP 到达后，查询实际姿态 → 决定是否推扫。"""
        # 查询并保存实际 gripper 姿态, APPROACH 将沿用此姿态实现纯平移
        self._pre_grasp_orientation = self._lookup_gripper_orientation()

        from .push_sweep import should_push_sweep

        obstacle = self._target_metadata.get("obstacle_above", "none")
        material = self._target_metadata.get("material", "soft")

        if should_push_sweep(obstacle, material):
            self.get_logger().info(
                f"检测到软障碍物: obstacle={obstacle}, material={material} → 执行推扫"
            )
            self._transition(self.PUSH_SWEEP)
        else:
            self.get_logger().info(
                f"无需推扫: obstacle={obstacle}, material={material} → 直接 APPROACH"
            )
            self._transition(self.APPROACH)

    # ── 推扫排障 ──────────────────────────────────

    def _do_push_sweep(self):
        """执行推扫排障轨迹。"""
        from .push_sweep import generate_sweep_waypoints

        if self._pre_grasp_target is None:
            self.get_logger().error("PUSH_SWEEP: _pre_grasp_target is None, 降级为 APPROACH")
            self._transition(self.APPROACH)
            return

        direction = self._target_metadata.get(
            "optimal_approach_direction", "top"
        )

        z_offset = self.get_parameter("pre_grasp_z_offset").value
        pre_grasp_pose = Pose()
        pre_grasp_pose.position = Point(
            x=self._pre_grasp_target.pose.position.x,
            y=self._pre_grasp_target.pose.position.y,
            z=self._pre_grasp_target.pose.position.z + z_offset,
        )

        waypoints = generate_sweep_waypoints(pre_grasp_pose, direction)
        if not waypoints:
            self.get_logger().info("推扫方向无需推扫, 跳过")
            self._transition(self.APPROACH)
            return

        self.get_logger().info(
            f"PUSH_SWEEP: {len(waypoints)} 个航点, direction={direction}"
        )
        self._execute_sweep_trajectory(waypoints)

    def _execute_sweep_trajectory(self, waypoints):
        """通过 Cartesian path 规划并执行推扫轨迹。"""
        if not self._cartesian_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("compute_cartesian_path 不可用 (推扫)")
            self._transition(self.APPROACH)
            return

        req = GetCartesianPath.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.header.frame_id = "world"
        req.group_name = "arm"
        req.link_name = "gripper"
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self._cartesian_cli.call_async(req)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if future.done():
                break
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            self.get_logger().error("compute_cartesian_path 超时 (推扫)")
            self._transition(self.APPROACH)
            return

        resp = future.result()
        if resp is None or resp.error_code.val != 1:
            self.get_logger().warning(
                f"推扫 Cartesian 规划失败, 降级为直接 APPROACH。 "
                f"fraction={resp.fraction if resp else 0.0:.2f}"
            )
            self._transition(self.APPROACH)
            return

        self.get_logger().info(
            f"推扫 Cartesian 规划成功, fraction={resp.fraction:.2f}"
        )
        self._sweep_in_progress = True
        self._execute_cartesian_trajectory(resp.solution.joint_trajectory)

    # ── 接近 ──────────────────────────────────────

    def _do_approach(self):
        """张爪 + 规划下移。"""
        self.get_logger().info("APPROACH: 张爪...")
        self._send_gripper_command(
            self.get_parameter("gripper_open_pos").value,
            done_callback=self._on_approach_open_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_approach_open_done(self):
        approach_mode = self.get_parameter("approach_mode").value
        self.get_logger().info(f"APPROACH: 碰撞策略 ({approach_mode})...")

        # Cartesian 下探前移除物体, 允许夹爪进入目标空间
        if self._target_object_registered:
            self._psm.remove_object("target")
            self._target_object_registered = False
            self.get_logger().info("APPROACH: 物体已移除, 准备 Cartesian 下探")
        self._plan_approach()

    def _plan_approach(self):
        """Cartesian 直线下移——使用 P_rough + jaw_clearance 作为目标。"""
        self.get_logger().info("规划 Cartesian 接近路径...")
        if self._pre_grasp_target is None:
            self._transition(self.FAILED)
            return

        if not self._cartesian_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("compute_cartesian_path 不可用")
            self._transition(self.FAILED)
            return

        jaw_clearance = self.get_parameter("jaw_clearance").value

        # 夹爪 TCP 到指尖的实际长度 (需根据 URDF 微调)
        finger_length_offset = 0.05
        # 期望指尖包住物体上表面的深度
        grasp_depth = 0.02

        # P_rough = 物体上表面 (深度相机首击点)
        # TCP 目标 Z = 上表面 + 指尖长度补偿 - 抓取深度 + 夹爪几何补偿
        # 这样指尖到达上表面下方 grasp_depth 处，TCP 停在上表面上方安全位置
        target_z = (self._pre_grasp_target.pose.position.z
                    + finger_length_offset
                    - grasp_depth
                    + jaw_clearance)

        approach_waypoint = Pose()
        approach_waypoint.position = Point(
            x=self._pre_grasp_target.pose.position.x,
            y=self._pre_grasp_target.pose.position.y,
            z=target_z,
        )
        approach_waypoint.orientation = (
            self._pre_grasp_orientation
            if self._pre_grasp_orientation is not None
            else Quaternion(w=1.0)
        )
        self.get_logger().info(
            f"APPROACH waypoint: pos=({approach_waypoint.position.x:.3f}, "
            f"{approach_waypoint.position.y:.3f}, {approach_waypoint.position.z:.3f})"
            f" (P_rough={self._pre_grasp_target.pose.position.z:.3f}"
            f" + finger={finger_length_offset:.3f} - grasp={grasp_depth:.3f}"
            f" + jaw={jaw_clearance:.3f})"
        )

        req = GetCartesianPath.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.header.frame_id = self._pre_grasp_target.header.frame_id or "world"
        req.group_name = "arm"
        req.link_name = "gripper"
        req.waypoints = [approach_waypoint]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self._cartesian_cli.call_async(req)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if future.done():
                break
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            self.get_logger().error("compute_cartesian_path 超时")
            self._transition(self.FAILED)
            return

        resp = future.result()
        if resp is None or resp.error_code.val != 1:
            frac = resp.fraction if resp else 0.0
            self.get_logger().error(
                f"Cartesian 路径规划失败, fraction={frac:.2f}, "
                f"error_code={resp.error_code.val if resp else 'N/A'}"
            )
            if frac < 0.9:
                self._transition(self.FAILED)
                return
            self.get_logger().warning(
                f"Cartesian 路径不完整 (fraction={frac:.2f}), 尝试执行可达部分"
            )

        self.get_logger().info(f"Cartesian 路径规划成功, fraction={resp.fraction:.2f}")
        self._execute_cartesian_trajectory(resp.solution.joint_trajectory)

    def _execute_cartesian_trajectory(self, joint_trajectory):
        if not self._arm_traj_action.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("arm_controller action 不可用")
            self._transition(self.FAILED)
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory
        send_future = self._arm_traj_action.send_goal_async(goal)
        send_future.add_done_callback(self._on_cartesian_exec_done)

    def _on_cartesian_exec_done(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"发送 Cartesian 轨迹失败: {exc}")
            self._transition(self.FAILED)
            return
        if not goal_handle.accepted:
            self.get_logger().warning("arm_controller 拒绝 Cartesian 轨迹")
            self._transition(self.FAILED)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_cartesian_result)

    def _on_cartesian_result(self, future):
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f"Cartesian 执行失败: {exc}")
            self._transition(self.FAILED)
            return
        if wrapped.result.error_code == 0:
            self.get_logger().info("Cartesian 轨迹完成")
            if self._sweep_in_progress:
                self._sweep_in_progress = False
                self.get_logger().info("推扫完成, 进入 APPROACH")
                self._transition(self.APPROACH)
            else:
                self._on_approach_done()
        else:
            self.get_logger().warning(f"Cartesian 执行失败: {wrapped.result.error_string}")
            if self._sweep_in_progress:
                self._sweep_in_progress = False
                self.get_logger().warning("推扫执行失败, 降级为直接 APPROACH")
                self._transition(self.APPROACH)
            else:
                self._transition(self.FAILED)

    def _on_approach_done(self):
        self.get_logger().info("到达抓取位姿")
        self._transition(self.GRASP)

    # ── 抓取 ──────────────────────────────────────

    def _do_grasp(self):
        self.get_logger().info("闭合夹爪...")
        self._send_gripper_command(
            self.get_parameter("gripper_close_pos").value,
            done_callback=self._on_gripper_close_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_gripper_close_done(self):
        self.get_logger().info("双层附着: Gazebo AttachLink + MoveIt attach_object...")

        if not self._attach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("AttachLink 服务不可用")
            self._transition(self.FAILED)
            return

        req = AttachLink.Request()
        req.model1_name = "so101"
        req.link1_name = "gripper"
        req.model2_name = "target_box"
        req.link2_name = "box_link"

        future = self._attach_cli.call_async(req)
        future.add_done_callback(self._on_attach_done)

    def _on_attach_done(self, future):
        try:
            resp = future.result()
            if not resp.success:
                self.get_logger().error(f"AttachLink 失败: {resp.message}")
                self._transition(self.FAILED)
                return
            self.get_logger().info("Gazebo AttachLink 成功")
        except Exception as exc:
            self.get_logger().error(f"AttachLink 异常: {exc}")
            self._transition(self.FAILED)
            return

        # MoveIt 层附着
        self._psm.attach_object("target", "gripper", ["jaw"])
        self.get_logger().info("MoveIt attach_object 完成")

        self._transition(self.LIFT_PLAN)

    # ── 抬起 ──────────────────────────────────────

    def _plan_lift(self):
        self.get_logger().info("规划抬起 (物体已附着, 自动防碰)...")
        lift_pose = Pose()
        lift_pose.position = Point(
            x=self._pre_grasp_target.pose.position.x,
            y=self._pre_grasp_target.pose.position.y,
            z=self._pre_grasp_target.pose.position.z
            + self.get_parameter("pre_grasp_z_offset").value,
        )
        lift_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            lift_pose, "world",
            success_callback=lambda: self._transition(self.TRANSPORT_PLAN),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 搬运 ──────────────────────────────────────

    def _plan_transport(self):
        self.get_logger().info("规划搬运...")
        placement = self.get_parameter("placement_position").value
        place_pose = Pose()
        place_pose.position = Point(x=placement[0], y=placement[1], z=placement[2])
        place_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            place_pose, "world",
            success_callback=lambda: self._transition(self.PLACE),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 放置 ──────────────────────────────────────

    def _do_place(self):
        self.get_logger().info("PLACE: 双层脱离...")

        if self._detach_cli.wait_for_service(timeout_sec=2.0):
            req = DetachLink.Request()
            req.model1_name = "so101"
            req.link1_name = "gripper"
            req.model2_name = "target_box"
            req.link2_name = "box_link"
            future = self._detach_cli.call_async(req)
            future.add_done_callback(self._on_detach_done)
            return
        else:
            self.get_logger().warning("DetachLink 不可用")
            self._on_detach_done(None)

    def _on_detach_done(self, future):
        if future is not None:
            try:
                resp = future.result()
                if resp.success:
                    self.get_logger().info("Gazebo DetachLink 成功")
            except Exception as exc:
                self.get_logger().warning(f"DetachLink: {exc}")

        # MoveIt 层清理: detach + forbid + remove
        self._psm.detach_object("target", "gripper")
        self._psm.forbid_collision("gripper", "target")

        if self._target_object_registered:
            self._psm.remove_object("target")

        # 张爪
        self._send_gripper_command(
            self.get_parameter("gripper_open_pos").value,
            done_callback=self._on_place_complete,
            fail_callback=self._on_place_complete,
        )

    def _on_place_complete(self):
        self.get_logger().info("抓取流程完成!")
        self._publish_status("grasp_complete")
        self._state = self.IDLE

    # ── MoveGroup 封装 ────────────────────────────

    def _send_move_request(self, target_pose, frame_id, *,
                           success_callback, fail_callback,
                           use_orientation=False,
                           pos_tolerance=0.01, ori_tolerance=0.1):
        goal = MoveGroup.Goal()
        goal.request = self._build_request(
            target_pose, frame_id,
            use_orientation=use_orientation,
            pos_tolerance=pos_tolerance,
            ori_tolerance=ori_tolerance,
        )
        goal.planning_options = PlanningOptions()

        send_future = self._move_action.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._move_response_cb(f, success_callback, fail_callback)
        )

    def _build_request(self, pose, frame_id, *,
                       use_orientation=False,
                       pos_tolerance=0.01, ori_tolerance=0.1):
        req = MotionPlanRequest()
        req.group_name = "arm"
        req.num_planning_attempts = 10
        req.allowed_planning_time = self.get_parameter("planning_time").value
        req.max_velocity_scaling_factor = self.get_parameter("velocity_scaling").value
        req.max_acceleration_scaling_factor = self.get_parameter("velocity_scaling").value

        constraints = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = "gripper"
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tolerance]

        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(pose)
        pc.constraint_region = region
        constraints.position_constraints.append(pc)

        if use_orientation:
            oc = OrientationConstraint()
            oc.header.frame_id = frame_id
            oc.link_name = "gripper"
            oc.orientation = pose.orientation
            # 仅约束 X 轴(接近方向), Y/Z 放松 — 5-DOF 臂解不了 6 DOF 全约束
            oc.absolute_x_axis_tolerance = ori_tolerance
            oc.absolute_y_axis_tolerance = 3.0   # 几乎不约束 jaw 开口方向
            oc.absolute_z_axis_tolerance = 3.0   # 几乎不约束剩余轴
            oc.weight = 1.0
            constraints.orientation_constraints.append(oc)

        req.goal_constraints.append(constraints)
        return req

    def _move_response_cb(self, future, success_cb, fail_cb):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"发送 MoveGroup 目标失败: {exc}")
            fail_cb()
            return

        if not goal_handle.accepted:
            self.get_logger().warning("MoveGroup 拒绝目标")
            fail_cb()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._move_result_cb(f, success_cb, fail_cb)
        )

    def _move_result_cb(self, future, success_cb, fail_cb):
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f"获取 MoveGroup 结果失败: {exc}")
            fail_cb()
            return

        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"MoveGroup 成功, time={wrapped.result.planning_time:.2f}s")
            success_cb()
        else:
            self.get_logger().warning(
                f"MoveGroup 失败, status={wrapped.status}, "
                f"error={wrapped.result.error_code.val}"
            )
            fail_cb()

    # ── 夹爪控制 ──────────────────────────────────

    def _send_gripper_command(self, position, *, done_callback, fail_callback):
        if not self._gripper_action.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("夹爪 action server 不可用")
            fail_callback()
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["6"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start.sec = 1
        goal.trajectory.points = [pt]

        send_future = self._gripper_action.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._gripper_response_cb(f, done_callback, fail_callback)
        )

    def _gripper_response_cb(self, future, done_callback, fail_callback):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"发送夹爪指令失败: {exc}")
            fail_callback()
            return
        if not goal_handle.accepted:
            self.get_logger().warning("夹爪控制器拒绝指令")
            fail_callback()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._gripper_result_cb(f, done_callback, fail_callback)
        )

    def _gripper_result_cb(self, future, done_callback, fail_callback):
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f"获取夹爪结果失败: {exc}")
            fail_callback()
            return
        if wrapped.result.error_code == 0:
            done_callback()
        else:
            self.get_logger().warning(f"夹爪执行失败: {wrapped.result.error_string}")
            fail_callback()


def main(args=None):
    rclpy.init(args=args)
    node = GrabActionSimple()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
