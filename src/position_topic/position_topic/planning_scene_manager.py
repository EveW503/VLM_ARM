"""
Planning Scene 操作封装。

通过 MoveIt 监听的 topic 操作世界模型 (零阻塞):
- /collision_object: 碰撞物体增删
- /attached_collision_object: 物体附着/脱离

注意: allow_collision/forbid_collision 需要 ApplyPlanningScene service，
当前通过 remove_object 替代实现。
"""
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive


class PlanningSceneManager:
    """管理 MoveIt Planning Scene 中的物体、碰撞、附着 (topic 方式, 零阻塞)。"""

    def __init__(self, node: Node):
        self._logger = node.get_logger()
        self._collision_pub = node.create_publisher(
            CollisionObject, "/collision_object", 10
        )
        self._attached_pub = node.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", 10
        )

    # ── 物体管理 ──────────────────────────────────

    def add_object(self, object_id, shape_type, dimensions, pose, frame_id):
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = frame_id
        obj.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = shape_type
        prim.dimensions = dimensions
        obj.primitives = [prim]
        obj.primitive_poses = [pose]

        self._collision_pub.publish(obj)
        self._logger.info(f"CollisionObject ADD: {object_id}")

    def remove_object(self, object_id):
        obj = CollisionObject()
        obj.id = object_id
        obj.operation = CollisionObject.REMOVE
        self._collision_pub.publish(obj)
        self._logger.info(f"CollisionObject REMOVE: {object_id}")

    def remove_all_objects(self):
        obj = CollisionObject()
        obj.id = ""
        obj.operation = CollisionObject.REMOVE
        self._collision_pub.publish(obj)
        self._logger.info("CollisionObject REMOVE all")

    # ── 附着/脱离 ────────────────────────────────

    def attach_object(self, object_id, link_name, touch_links=None):
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD
        aco.link_name = link_name
        if touch_links:
            aco.touch_links = touch_links
        aco.weight = 1.0

        self._attached_pub.publish(aco)
        self._logger.info(f"AttachedCollisionObject ADD: {object_id} → {link_name}")

    def detach_object(self, object_id, link_name):
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        aco.link_name = link_name

        self._attached_pub.publish(aco)
        self._logger.info(f"AttachedCollisionObject REMOVE: {object_id}")

    # ── 碰撞策略 (简化版: 通过 remove 替代) ─────

    def allow_collision(self, link1, object_id):
        """topic 方式不支持 ACM 操作, 通过 remove_object 让 MoveIt 不避让。"""
        self.remove_object(object_id)

    def forbid_collision(self, link1, object_id):
        """topic 方式不支持 ACM 操作, 此处为 no-op。"""
        pass
