"""
Planning Scene 操作封装。

通过 moveit_msgs/srv/ApplyPlanningScene 服务操作 MoveIt 的世界模型:
- 碰撞物体管理 (add/remove)
- 碰撞策略 (allow/forbid collision pairs)
- 物体附着/脱离 (attach/detach)
"""
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene
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

    # ── 物体管理 ──────────────────────────────────

    def add_object(self, object_id, shape_type, dimensions, pose, frame_id):
        """
        向 Planning Scene 添加碰撞物体。
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
        obj.id = ""
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
        scene = PlanningScene()
        scene.is_diff = True

        entry = AllowedCollisionEntry()
        entry.enabled = [False]
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

        # time.sleep 循环等结果，不碰 executor API
        # MultiThreadedExecutor 的其他线程自然处理 service 响应
        import time
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if future.done():
                break
            time.sleep(0.05)

        if future.done():
            if future.result() is not None:
                if future.result().success:
                    return True
                self._logger.warning(f"ApplyPlanningScene 失败 ({tag})")
                return False
            exc = future.exception()
            if exc:
                self._logger.warning(f"ApplyPlanningScene 异常 ({tag}): {exc}")
                return False
        self._logger.warning(f"ApplyPlanningScene 超时 ({tag})")
        return False
