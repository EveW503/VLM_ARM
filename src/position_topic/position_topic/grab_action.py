"""
Grab Action 节点: 抓取全流程状态机。

订阅 /target_pre_grasp (阶段一粗定位) 和 /target_pose (阶段二精定位),
执行: 靠近→下移→夹取→抬起→搬运→放置→释放。
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume, Constraints, MotionPlanRequest,
    PlanningOptions, PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from linkattacher_msgs.srv import AttachLink, DetachLink
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GrabAction(Node):
    # 状态常量
    IDLE = "idle"
    PRE_GRASP_PLAN = "pre_grasp_plan"
    PRE_GRASP_EXEC = "pre_grasp_exec"
    AWAIT_STAGE2 = "await_stage2"
    APPROACH_PLAN = "approach_plan"
    APPROACH_EXEC = "approach_exec"
    GRASP = "grasp"
    LIFT_PLAN = "lift_plan"
    LIFT_EXEC = "lift_exec"
    TRANSPORT_PLAN = "transport_plan"
    TRANSPORT_EXEC = "transport_exec"
    PLACE = "place"
    FAILED = "failed"

    def __init__(self):
        super().__init__("grab_action")

        # --- 参数 ---
        self.declare_parameter("placement_position", [0.2, -0.15, 0.25])
        self.declare_parameter("pre_grasp_z_offset", 0.08)
        self.declare_parameter("approach_step", 0.01)
        self.declare_parameter("gripper_close_pos", -0.17)
        self.declare_parameter("gripper_open_pos", 0.5)
        self.declare_parameter("stage2_timeout", 10.0)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.3)

        # --- 输入订阅 ---
        self.create_subscription(
            PoseStamped, "/target_pre_grasp", self._pre_grasp_cb, 10
        )
        self.create_subscription(
            PoseStamped, "/target_pose", self._target_pose_cb, 10
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

        # --- 状态机 ---
        self._state = self.IDLE
        self._pre_grasp_target = None
        self._grasp_target = None
        self._stage2_timer = None

        self.get_logger().info("Grab Action 节点已启动")

    # ── 回调 ──────────────────────────────────────────

    def _pre_grasp_cb(self, msg):
        if self._state == self.IDLE:
            self.get_logger().info(
                f"收到预抓取目标: ({msg.pose.position.x:.3f}, "
                f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
            )
            self._pre_grasp_target = msg
            self._grasp_target = None
            self._transition(self.PRE_GRASP_PLAN)

    def _target_pose_cb(self, msg):
        self.get_logger().info(
            f"收到精确抓取位姿: ({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )
        self._grasp_target = msg
        if self._state == self.AWAIT_STAGE2:
            if self._stage2_timer is not None:
                self.destroy_timer(self._stage2_timer)
                self._stage2_timer = None
            self._transition(self.APPROACH_PLAN)

    # ── 状态机驱动 ────────────────────────────────────

    def _transition(self, new_state):
        self._state = new_state
        if new_state == self.PRE_GRASP_PLAN:
            self._plan_pre_grasp()
        elif new_state == self.APPROACH_PLAN:
            self._plan_approach()
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
            self._publish_status("error")
            self._state = self.IDLE

    def _publish_status(self, status):
        msg = String(data=status)
        self._status_pub.publish(msg)

    # ── 预抓取 ──────────────────────────────────────

    def _plan_pre_grasp(self):
        self.get_logger().info("规划预抓取位姿...")
        target = self._pre_grasp_target
        if target is None:
            self._transition(self.FAILED)
            return

        pre_grasp_pose = Pose()
        pre_grasp_pose.position = Point(
            x=target.pose.position.x,
            y=target.pose.position.y,
            z=target.pose.position.z + self.get_parameter("pre_grasp_z_offset").value,
        )
        pre_grasp_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            pre_grasp_pose, "base",
            success_callback=self._on_pre_grasp_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_pre_grasp_done(self):
        self.get_logger().info("预抓取完成, 等待阶段二精定位...")
        self._publish_status("stage1_done")
        self._state = self.AWAIT_STAGE2
        self._stage2_timer = self.create_timer(
            self.get_parameter("stage2_timeout").value,
            self._stage2_timeout_cb,
        )

    def _stage2_timeout_cb(self):
        if self._state != self.AWAIT_STAGE2:
            return
        self.get_logger().warning("阶段二超时, 使用降级策略")
        self.destroy_timer(self._stage2_timer)
        self._stage2_timer = None
        if self._grasp_target is None:
            t = self._pre_grasp_target
            fb_z = t.pose.position.z - self.get_parameter("pre_grasp_z_offset").value
            self._grasp_target = PoseStamped()
            self._grasp_target.header.frame_id = "base"
            self._grasp_target.pose = Pose(
                position=Point(x=t.pose.position.x, y=t.pose.position.y, z=fb_z),
                orientation=Quaternion(w=1.0),
            )
        self._transition(self.APPROACH_PLAN)

    # ── 接近 (直线下移) ────────────────────────────

    def _plan_approach(self):
        self.get_logger().info("规划接近路径...")
        if self._grasp_target is None:
            self._transition(self.FAILED)
            return

        self._send_move_request(
            self._grasp_target.pose, self._grasp_target.header.frame_id or "base",
            success_callback=self._on_approach_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

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
        self.get_logger().info("调用 AttachLink...")

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
            if resp.success:
                self.get_logger().info("物体已附着")
                self._publish_status("stage2_done")
                self._transition(self.LIFT_PLAN)
            else:
                self.get_logger().error(f"AttachLink 失败: {resp.message}")
                self._transition(self.FAILED)
        except Exception as exc:
            self.get_logger().error(f"AttachLink 调用异常: {exc}")
            self._transition(self.FAILED)

    # ── 抬起 ──────────────────────────────────────

    def _plan_lift(self):
        self.get_logger().info("规划抬起路径...")
        lift_pose = Pose()
        lift_pose.position = Point(
            x=self._pre_grasp_target.pose.position.x,
            y=self._pre_grasp_target.pose.position.y,
            z=self._pre_grasp_target.pose.position.z
            + self.get_parameter("pre_grasp_z_offset").value,
        )
        lift_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            lift_pose, "base",
            success_callback=lambda: self._transition(self.TRANSPORT_PLAN),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 搬运 ──────────────────────────────────────

    def _plan_transport(self):
        self.get_logger().info("规划搬运到放置位姿...")
        placement = self.get_parameter("placement_position").value
        place_pose = Pose()
        place_pose.position = Point(
            x=placement[0], y=placement[1], z=placement[2]
        )
        place_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            place_pose, "base",
            success_callback=lambda: self._transition(self.PLACE),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 放置 ──────────────────────────────────────

    def _do_place(self):
        self.get_logger().info("释放物体...")

        if not self._detach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warning("DetachLink 服务不可用, 仍然张开夹爪")
        else:
            req = DetachLink.Request()
            req.model1_name = "so101"
            req.link1_name = "gripper"
            req.model2_name = "target_box"
            req.link2_name = "box_link"
            future = self._detach_cli.call_async(req)
            future.add_done_callback(self._on_detach_done)
            return

        self._send_gripper_command(
            self.get_parameter("gripper_open_pos").value,
            done_callback=self._on_place_complete,
            fail_callback=self._on_place_complete,
        )

    def _on_detach_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info("物体已释放")
            else:
                self.get_logger().warning(f"DetachLink: {resp.message}")
        except Exception as exc:
            self.get_logger().warning(f"DetachLink 调用异常: {exc}")

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
                           success_callback, fail_callback):
        goal = MoveGroup.Goal()
        goal.request = self._build_request(target_pose, frame_id)
        goal.planning_options = PlanningOptions()

        send_future = self._move_action.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._move_response_cb(f, success_callback, fail_callback)
        )

    def _build_request(self, pose, frame_id):
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
        sphere.dimensions = [0.01]

        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(pose)
        pc.constraint_region = region
        constraints.position_constraints.append(pc)
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
            fail_cb()
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

    def _gripper_response_cb(self, future, done_cb, fail_cb):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"发送夹爪指令失败: {exc}")
            fail_cb()
            return
        if not goal_handle.accepted:
            self.get_logger().warning("夹爪控制器拒绝指令")
            fail_cb()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._gripper_result_cb(f, done_cb, fail_cb)
        )

    def _gripper_result_cb(self, future, done_cb, fail_cb):
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f"获取夹爪结果失败: {exc}")
            fail_cb()
            return
        if wrapped.result.error_code == 0:
            done_cb()
        else:
            self.get_logger().warning(f"夹爪执行失败: {wrapped.result.error_string}")
            fail_cb()


def main(args=None):
    rclpy.init(args=args)
    node = GrabAction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
