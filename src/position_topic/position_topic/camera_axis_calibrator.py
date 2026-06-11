"""
相机轴标定节点: 发布一系列已知 gripper 朝向, 在 Gazebo 中观察 optical frame 轴颜色。
用法: 启动后观察 Gazebo 中 ee_camera_optical_link 的红/绿/蓝轴指向,
      记录每个测试姿态下各轴对应的世界方向。
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


class CameraAxisCalibrator(Node):
    def __init__(self):
        super().__init__("camera_axis_calibrator")
        self._pub = self.create_publisher(PoseStamped, "/target_observation_pose", 10)
        self._idx = 0

        # 所有测试姿态用同一位置 (臂可达的安全位置), 仅改变朝向
        self._base_pos = (0.20, 0.0, 0.25)

        # 依次测试: identity, 绕各轴旋转
        self._tests = [
            # (label, qx, qy, qz, qw)
            ("identity",       0.0, 0.0, 0.0, 1.0),
            ("Rx(+90°)",       0.7071, 0.0, 0.0, 0.7071),
            ("Rx(-90°)",      -0.7071, 0.0, 0.0, 0.7071),
            ("Ry(+90°)",       0.0, 0.7071, 0.0, 0.7071),
            ("Ry(-90°)",       0.0, -0.7071, 0.0, 0.7071),
            ("Rz(+90°)",       0.0, 0.0, 0.7071, 0.7071),
            ("Rz(-90°)",       0.0, 0.0, -0.7071, 0.7071),
            # 当前 PRE_GRASP 朝向
            ("PRE_GRASP_Ry(-90°)", 0.0, -0.7071, 0.0, 0.7071),
        ]

        self._timer = self.create_timer(5.0, self._publish_next)
        self.get_logger().info("相机轴标定节点已启动, 每 5 秒切换一次朝向...")

    def _publish_next(self):
        if self._idx >= len(self._tests):
            self.get_logger().info("所有测试完成. 按 Ctrl-C 退出.")
            self._timer.cancel()
            return

        label, qx, qy, qz, qw = self._tests[self._idx]
        msg = PoseStamped()
        msg.header.frame_id = "base"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = Pose(
            position=Point(x=self._base_pos[0], y=self._base_pos[1], z=self._base_pos[2]),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
        self._pub.publish(msg)
        self.get_logger().info(
            f"[{self._idx+1}/{len(self._tests)}] 已发布: {label} "
            f"q=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})"
        )
        self._idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = CameraAxisCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
