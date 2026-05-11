# Grab Action Planning Scene 改造 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 `moveit_msgs.srv.ApplyPlanningScene` 服务重写抓取管线，让 MoveIt Planning Scene 感知目标物体并在抓取全程中正确管理碰撞、附着、脱离。

**Architecture:** 新建 `planning_scene_manager.py` 封装 Planning Scene 操作（add/remove/attach/detach/allow_forbid），重写 `grab_action.py` 新增 SETUP 阶段、将张爪移至 APPROACH、双层同步（Gazebo LinkAttacher + MoveIt PlanningScene）。

**Tech Stack:** ROS 2 Humble, moveit_msgs.srv.ApplyPlanningScene, moveit_msgs.srv.GetPlanningScene, SolidPrimitive, CollisionObject, AttachedCollisionObject

---

## 文件映射

```
position_topic/position_topic/
├── planning_scene_manager.py  ← 新建: Planning Scene 操作封装
└── grab_action.py             ← 重写: 集成 PSM + 新状态机

position_topic/
└── setup.py                   ← 不变 (entry_points 已有 grab_action)
```

---

### Task 1: 创建 planning_scene_manager.py

**Files:**
- Create: `position_topic/position_topic/planning_scene_manager.py`

- [ ] **Step 1: 创建文件**

```python
"""
Planning Scene 操作封装。

通过 moveit_msgs/srv/ApplyPlanningScene 服务操作 MoveIt 的世界模型:
- 碰撞物体管理 (add/remove)
- 碰撞策略 (allow/forbid collision pairs)
- 物体附着/脱离 (attach/detach)
"""
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_msgs.msg import (
    PlanningScene, CollisionObject, AttachedCollisionObject,
    AllowedCollisionEntry,
)
from shape_msgs.msg import SolidPrimitive


class PlanningSceneManager:
    """管理 MoveIt Planning Scene 中的物体、碰撞、附着。"""

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()
        self._apply_cli = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._get_cli = node.create_client(GetPlanningScene, "/get_planning_scene")

    # ── 物体管理 ──────────────────────────────────

    def add_object(self, object_id, shape_type, dimensions, pose, frame_id):
        """
        向 Planning Scene 添加碰撞物体。

        Args:
            object_id: str, 物体标识符
            shape_type: int, SolidPrimitive.BOX / SPHERE / CYLINDER
            dimensions: list[float], 尺寸 [x,y,z] 或 [radius,height]
            pose: Pose, 物体位姿
            frame_id: str, 位姿参考坐标系

        Returns:
            bool: 成功返回 True
        """
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = frame_id
        obj.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = shape_type
        prim.dimensions = dimensions
        obj.primitives = [prim]
        obj.primitive_poses = [pose]

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        return self._call_apply(scene, f"add_object({object_id})")

    def remove_object(self, object_id):
        obj = CollisionObject()
        obj.id = object_id
        obj.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        return self._call_apply(scene, f"remove_object({object_id})")

    def remove_all_objects(self):
        obj = CollisionObject()
        obj.id = "__all__"
        obj.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        return self._call_apply(scene, "remove_all_objects")

    # ── 碰撞策略 ──────────────────────────────────

    def allow_collision(self, link1, object_id):
        """
        允许特定 link 与物体碰撞（如夹爪接触目标时）。
        """
        scene = PlanningScene()
        scene.is_diff = True

        entry = AllowedCollisionEntry()
        entry.enabled = [True]
        scene.allowed_collision_matrix.entry_names = [link1, object_id]
        scene.allowed_collision_matrix.entry_values = [entry]

        return self._call_apply(scene, f"allow_collision({link1}, {object_id})")

    def forbid_collision(self, link1, object_id):
        entry = AllowedCollisionEntry()
        entry.enabled = [False]
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix.entry_names = [link1, object_id]
        scene.allowed_collision_matrix.entry_values = [entry]

        return self._call_apply(scene, f"forbid_collision({link1}, {object_id})")

    # ── 附着/脱离 ────────────────────────────────

    def attach_object(self, object_id, link_name, touch_links=None):
        """
        将物体附着到机器人 link（MoveIt 层，与 Gazebo LinkAttacher 平行）。
        """
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD
        aco.link_name = link_name
        if touch_links:
            aco.touch_links = touch_links
        aco.weight = 1.0

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]

        return self._call_apply(scene, f"attach_object({object_id})")

    def detach_object(self, object_id, link_name):
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        aco.link_name = link_name

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]

        return self._call_apply(scene, f"detach_object({object_id})")

    # ── 内部 ──────────────────────────────────────

    def _call_apply(self, scene, tag):
        if not self._apply_cli.wait_for_service(timeout_sec=2.0):
            self._logger.warning(f"ApplyPlanningScene 不可用 ({tag})")
            return False

        req = ApplyPlanningScene.Request()
        req.scene = scene

        future = self._apply_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)

        if future.done() and future.result() is not None:
            if future.result().success:
                return True
            self._logger.warning(f"ApplyPlanningScene 失败 ({tag})")
        else:
            self._logger.warning(f"ApplyPlanningScene 超时 ({tag})")
        return False
```

- [ ] **Step 2: 验证语法 + 导入**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import py_compile; py_compile.compile('src/position_topic/position_topic/planning_scene_manager.py', doraise=True); print('OK')"`

Expected: `OK`

- [ ] **Step 3: 验证模块可导入**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import sys; sys.path.insert(0, 'src/position_topic'); from position_topic.planning_scene_manager import PlanningSceneManager; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/position_topic/position_topic/planning_scene_manager.py
git commit -m "feat: add PlanningSceneManager for collision object and grasp management"
```

---

### Task 2: 重写 grab_action.py

**Files:**
- Modify: `position_topic/position_topic/grab_action.py` (重写)

- [ ] **Step 1: 重写完整节点**

```python
"""
Grab Action 节点: 抓取全流程状态机。

集成 PlanningSceneManager 管理 MoveIt 世界模型:
- SETUP: 注册目标物体到 planning scene
- PRE_GRASP: 物体在 scene → MoveIt 自动绕行
- APPROACH: 张爪 + allow_collision → 进入目标
- GRASP: 双层附着 (Gazebo LinkAttacher + MoveIt attach_object)
- LIFT/TRANSPORT: 物体附着 → 自动防碰
- PLACE: 双层脱离 + 清理 scene

订阅 /target_pre_grasp (阶段一粗定位) 和 /target_pose (阶段二精定位).
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

from .planning_scene_manager import PlanningSceneManager


class GrabAction(Node):
    # 状态常量
    IDLE = "idle"
    SETUP = "setup"
    PRE_GRASP_PLAN = "pre_grasp_plan"
    AWAIT_STAGE2 = "await_stage2"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT_PLAN = "lift_plan"
    TRANSPORT_PLAN = "transport_plan"
    PLACE = "place"
    FAILED = "failed"

    def __init__(self):
        super().__init__("grab_action")

        # --- 参数 ---
        self.declare_parameter("placement_position", [0.2, -0.15, 0.25])
        self.declare_parameter("pre_grasp_z_offset", 0.08)
        self.declare_parameter("gripper_close_pos", -0.17)
        self.declare_parameter("gripper_open_pos", 0.5)
        self.declare_parameter("stage2_timeout", 10.0)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.3)
        self.declare_parameter("approach_mode", "auto")
        self.declare_parameter("target_object_shape", "BOX")
        self.declare_parameter("target_object_dims", [0.03, 0.03, 0.03])

        # --- Planning Scene Manager ---
        self._psm = PlanningSceneManager(self)

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
        self._target_object_registered = False

        self.get_logger().info("Grab Action 节点已启动 (PSM 集成)")

    # ── 回调 ──────────────────────────────────────────

    def _pre_grasp_cb(self, msg):
        if self._state == self.IDLE:
            self.get_logger().info(
                f"收到预抓取目标: ({msg.pose.position.x:.3f}, "
                f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
            )
            self._pre_grasp_target = msg
            self._grasp_target = None
            self._target_object_registered = False
            self._transition(self.SETUP)

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
            self._transition(self.APPROACH)

    # ── 状态机驱动 ────────────────────────────────────

    def _transition(self, new_state):
        self._state = new_state
        if new_state == self.SETUP:
            self._do_setup()
        elif new_state == self.PRE_GRASP_PLAN:
            self._plan_pre_grasp()
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
            self._publish_status("error")
            self._state = self.IDLE

    def _publish_status(self, status):
        self._status_pub.publish(String(data=status))

    # ── SETUP ─────────────────────────────────────

    def _do_setup(self):
        """注册目标物体到 Planning Scene，触发 vlm_bridge stage 2。"""
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

        success = self._psm.add_object(
            "target", shape_type, dims,
            target.pose, target.header.frame_id or "base"
        )
        if success:
            self._target_object_registered = True
            self.get_logger().info("目标物体已注册到 Planning Scene")
        else:
            self.get_logger().warning("目标物体注册失败，将使用旧行为")

        # 触发 vlm_bridge stage 2
        self._publish_status("stage1_done")
        self._transition(self.PRE_GRASP_PLAN)

    # ── 预抓取 ──────────────────────────────────────

    def _plan_pre_grasp(self):
        self.get_logger().info("规划预抓取位姿 (物体在 scene → MoveIt 绕行)...")
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
        self._transition(self.APPROACH)

    # ── 接近 ──────────────────────────────────────

    def _do_approach(self):
        """张爪 + 碰撞策略 + 规划下移。"""
        self.get_logger().info("APPROACH: 张爪...")
        # 在离目标仅 8cm 时张爪，不会引入规划失败
        self._send_gripper_command(
            self.get_parameter("gripper_open_pos").value,
            done_callback=self._on_approach_open_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_approach_open_done(self):
        approach_mode = self.get_parameter("approach_mode").value
        self.get_logger().info(f"APPROACH: 碰撞策略 ({approach_mode})...")

        if approach_mode in ("allow", "auto"):
            ok = self._psm.allow_collision("gripper", "target")
            if ok:
                self.get_logger().info("allow_collision 成功")
            elif approach_mode == "auto":
                self.get_logger().info("allow_collision 失败, fallback → remove_object")
                self._psm.remove_object("target")
            else:
                self.get_logger().warning("allow_collision 失败")
        elif approach_mode == "remove":
            self._psm.remove_object("target")

        self._plan_approach()

    def _plan_approach(self):
        self.get_logger().info("规划接近路径 (物体不再遮挡)...")
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
        self.get_logger().info("双层附着: Gazebo AttachLink + MoveIt attach_object...")

        if not self._attach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("AttachLink 服务不可用")
            self._transition(self.FAILED)
            return

        # Gazebo 层附着
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

        self._publish_status("stage2_done")
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
            lift_pose, "base",
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
            place_pose, "base",
            success_callback=lambda: self._transition(self.PLACE),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 放置 ──────────────────────────────────────

    def _do_place(self):
        self.get_logger().info("PLACE: 双层脱离...")

        # Gazebo 层脱离
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

    # ── MoveGroup 封装 (保持不变) ────────────────

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

    # ── 夹爪控制 (保持不变) ──────────────────────

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
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import py_compile; py_compile.compile('src/position_topic/position_topic/grab_action.py', doraise=True); print('OK')"`

Expected: `OK`

- [ ] **Step 3: 全量构建**

Run: `cd /home/ljq/lerobot_ws && colcon build --symlink-install --packages-select position_topic`

Expected: `Summary: 1 package finished`

- [ ] **Step 4: 验证导入链路**

Run: `source /home/ljq/lerobot_ws/install/setup.bash && timeout 5 python3 -c "
from position_topic.planning_scene_manager import PlanningSceneManager
from position_topic.grab_action import GrabAction
print('ALL IMPORTS OK')
" 2>&1`

Expected: `ALL IMPORTS OK`

- [ ] **Step 5: Commit**

```bash
git add src/position_topic/position_topic/grab_action.py
git commit -m "feat: rewrite grab_action with PlanningSceneManager integration

- Add SETUP stage: register target object in planning scene
- Move gripper open to APPROACH (closer to target, less sweep volume)
- Double-layer sync: Gazebo LinkAttacher + MoveIt attach_object
- Approach collision strategy: allow_collision with remove fallback
- PLACE: double-layer detach + scene cleanup"
```

---

### Task 3: 集成测试

**Files:** None (手动测试)

- [ ] **Step 1: 启动仿真**

```bash
source ~/lerobot_ws/install/setup.bash
ros2 launch position_topic test_no_strawberry.launch.py
```

Expected: 所有节点启动正常，无 import 错误

- [ ] **Step 2: 验证 Planning Scene 服务可用**

```bash
ros2 service list | grep planning_scene
```

Expected: `/apply_planning_scene` 和 `/get_planning_scene` 都在

- [ ] **Step 3: 测试硬编码抓取（跳过 VLM）**

```bash
ros2 topic pub /target_pre_grasp geometry_msgs/PoseStamped \
  "{header: {frame_id: 'base'}, pose: {position: {x: 0.25, y: 0.0, z: 0.08}}}" -1
```

观察日志关键点：
- `SETUP: 注册目标物体到 Planning Scene...` → `目标物体已注册`
- `预抓取完成 (物体在 scene → MoveIt 绕行)`
- `APPROACH: 张爪...` → `碰撞策略...` → `allow_collision 成功`
- `到达抓取位姿`
- `双层附着: Gazebo AttachLink + MoveIt attach_object...`
- `PLACE: 双层脱离...`
- `抓取流程完成!`

- [ ] **Step 4: 验证物体未被撞飞**

在 Gazebo GUI 中观察 target_box：预抓取时机械臂绕行（不过与 pre_grasp_z_offset 有关）、接近时直接到达物体上方、夹爪在物体旁闭合 → 物体不被弹飞。

- [ ] **Step 5: 验证搬运时物体附着**

物体被 attachment 后肉眼可见跟随机械臂移动 → 搬运到放置位姿 → 放置后脱落在放置位置。

- [ ] **Step 6: VLM 端到端（可选）**

```bash
ros2 topic pub /task_command std_msgs/String "data: '抓取红色方块'" -1
```

验证 VLM → vlm_bridge → grab_action 全链路。
