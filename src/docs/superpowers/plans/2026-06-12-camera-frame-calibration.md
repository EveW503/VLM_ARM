# Camera Coordinate Frame Calibration and Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 系统性确定 ee_camera 光学帧与 gripper 坐标系的真实映射关系，修复 `compute_lookat_quaternion` 和 `PRE_GRASP` 四元数，消除 ad-hoc 补偿。

**Architecture:** 创建 Gazebo 轴可视化测试节点 → 逐轴确定映射关系 → 修正 URDF 光学帧 RPY → 重写 `compute_lookat_quaternion`（无补偿 hack）→ 修正 PRE_GRASP 四元数 → 端到端验证。

**Tech Stack:** ROS 2 Humble, Gazebo Classic, Python 3, scipy.spatial.transform, TF2

---

## 背景

**问题本质**：`compute_lookat_quaternion` 当前使用 `y_axis = -direction` + `Ry(-π/2)` 补偿，但进入观察位姿后是红色轴（optical X）对目标，蓝色轴（optical Z/光轴）未对准。这个补偿是试错产物，底层原因是对 gripper → camera_link → optical_link 的轴映射关系没有完整确认。

**当前参数**：
- URDF camera mount on gripper: `origin_xyz="0.07 0.03 0.0"`, `origin_rpy="0 0 0"`
- Optical frame RPY: `3.1415927 0 -1.5707963` (π, 0, -π/2)
- 已知：X+ 使相机向 gripper 上方移动

**核心链路**：
```
gripper (基坐标系) → camera_link (identity RPY, 同 gripper 朝向)
  → ee_camera_optical_link (RPY = π, 0, -π/2)
```

`compute_lookat_quaternion` 返回的是 **gripper** 的目标朝向。MoveIt 将此朝向应用到 gripper link 上。要知道相机最终指向哪里，必须知道 optical frame 相对 gripper 的完整旋转变换。

---

### Task 1: 提交未提交的 error 处理改动

**Files:**
- Modify: `src/position_topic/position_topic/vlm_bridge.py:160-165` (已有改动)

- [ ] **Step 1: 确认当前 diff 内容**

```bash
cd ~/lerobot_ws/src
git diff position_topic/position_topic/vlm_bridge.py
```

确认改动仅为 `_grab_status_cb` 中新增的 `elif msg.data == "error"` 分支。

- [ ] **Step 2: 提交**

```bash
git add position_topic/position_topic/vlm_bridge.py
git commit -m "fix(vlm_bridge): add error handling for grab failure status"
```

---

### Task 2: 创建相机轴可视化诊断节点

**目的**：在 Gazebo 中发布一系列已知方向的 gripper 位姿，通过 Gazebo 的 link 坐标轴显示（红=X, 绿=Y, 蓝=Z）观察 optical frame 的实际朝向，确定真实轴映射。

**Files:**
- Create: `src/position_topic/position_topic/camera_axis_calibrator.py`
- Create: `src/position_topic/launch/camera_calibrate.launch.py`

- [ ] **Step 1: 创建标定节点**

```python
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
        # [0.20, 0.0, 0.25] 是已知安全点
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
```

- [ ] **Step 2: 注册到 setup.py entry_points**

```bash
# 检查当前 setup.py 的 entry_points 区块
grep -A 20 "entry_points" src/position_topic/setup.py
```

在 `console_scripts` 列表末尾添加：

```python
'camera_axis_calibrator = position_topic.camera_axis_calibrator:main',
```

- [ ] **Step 3: 创建标定用 launch 文件**

`src/position_topic/launch/camera_calibrate.launch.py`:

```python
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    gazebo_pkg = get_package_share_directory("lerobot_description")
    ctrl_pkg = get_package_share_directory("lerobot_controller")
    moveit_pkg = get_package_share_directory("lerobot_moveit")

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, "launch", "so101_gazebo.launch.py")
        )),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(ctrl_pkg, "launch", "so101_controller.launch.py")
        )),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, "launch", "so101_moveit.launch.py")
        )),
        Node(
            package="position_topic",
            executable="grab_action",
            name="grab_action",
            output="screen",
        ),
        Node(
            package="position_topic",
            executable="camera_axis_calibrator",
            name="camera_axis_calibrator",
            output="screen",
        ),
    ])
```

- [ ] **Step 4: 构建并确认可运行**

```bash
cd ~/lerobot_ws
colcon build --symlink-install --packages-select position_topic
source install/setup.bash
```

Expected: 构建成功，`ros2 run position_topic camera_axis_calibrator --help` 不报错。

- [ ] **Step 5: 提交**

```bash
git add src/position_topic/position_topic/camera_axis_calibrator.py \
        src/position_topic/launch/camera_calibrate.launch.py \
        src/position_topic/setup.py
git commit -m "feat: add camera axis calibration diagnostic node"
```

---

### Task 3: 执行标定并记录轴映射

> **注意**：此任务需要用户在 Gazebo 中目视观察，不能自动化。在 Gazebo 中启用 link 坐标系显示（View → Transparent → 勾选显示 link frames），然后观察 `ee_camera_optical_link` 的红(X)/绿(Y)/蓝(Z) 轴各指向哪个世界方向。

- [ ] **Step 1: 启动标定环境**

```bash
ros2 launch position_topic camera_calibrate.launch.py
```

- [ ] **Step 2: 在 Gazebo 中启用轴显示**

Gazebo 菜单 → View → 勾选 "Transparent" 或直接找到 `ee_camera_optical_link` →
右键 → "Follow" 或 "View" → 观察红/绿/蓝轴指向。

- [ ] **Step 3: 记录每个姿态的轴方向**

对每个测试姿态（identity, Rx(±90°), Ry(±90°), Rz(±90°), PRE_GRASP），记录下表：

| 测试姿态 | optical X (红) 指向 | optical Y (绿) 指向 | optical Z (蓝) 指向 |
|----------|-------------------|-------------------|-------------------|
| identity | (填写: 上/下/前/后/左/右) | | |
| Rx(+90°) | | | |
| Rx(-90°) | | | |
| Ry(+90°) | | | |
| Ry(-90°) | | | |
| Rz(+90°) | | | |
| Rz(-90°) | | | |

- [ ] **Step 4: 推导轴映射公式**

从 identity 测试可以直接得出：optical 各轴在 gripper (base) 坐标系中对应的基础方向。

设 gripper 的三个轴为 (Gx, Gy, Gz)，从 identity 测试得出：
- optical X (红) = ? Gripper 的哪个轴 (带正负号)
- optical Y (绿) = ? Gripper 的哪个轴
- optical Z (蓝) = ? Gripper 的哪个轴

这是修正 `compute_lookat_quaternion` 的核心输入。

- [ ] **Step 5: 更新内存文件**

将最终确认的轴映射写入 `/home/ljq/.claude/projects/-home-ljq-lerobot-ws/memory/so101-gripper-axis-mapping.md`，覆盖"已知未解决问题"中的旧记录。

---

### Task 4: 修正 compute_lookat_quaternion

**Files:**
- Modify: `src/position_topic/position_topic/camera_utils.py:166-204`

- [ ] **Step 1: 根据标定结果确定正确的方向轴映射**

基于 Task 3 的映射表，确定：在 gripper 坐标系中，哪个轴（正或负）映射到 optical Z（蓝色/光轴）。

设 optical Z = ±gripper 的某轴。在 `compute_lookat_quaternion` 中，我们构建的旋转矩阵是 **gripper 在世界中的朝向**，因此需要让这个方向指向目标。

- [ ] **Step 2: 重写 compute_lookat_quaternion（无补偿版）**

替换当前实现。以下是以 optical_Z = +gripper_X 为例的版本——实际轴映射需根据 Task 3 结果调整：

```python
def compute_lookat_quaternion(p_obs, p_target):
    """
    计算从观察点 p_obs 指向目标点 p_target 的四元数。
    使 ee_camera 的光轴 (optical Z, 蓝色) 对准目标方向。

    Args:
        p_obs: (3,) array-like, 观察点坐标 [x, y, z] (米, base frame)
        p_target: (3,) array-like, 目标点坐标 [x, y, z] (米, base frame)

    Returns:
        tuple: (qx, qy, qz, qw) 四元数 (gripper 朝向), 或 None
    """
    from scipy.spatial.transform import Rotation

    p_obs = np.asarray(p_obs, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)

    direction = p_target - p_obs
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None

    # 光轴方向: 从相机指向目标
    optical_z = direction / norm

    # world up = base frame Z+
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(optical_z, world_up)) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])

    # optical X = world_up × optical_z (归一化 → 位于水平面)
    optical_x = np.cross(world_up, optical_z)
    optical_x = optical_x / np.linalg.norm(optical_x)

    # optical Y = optical_z × optical_x (正交补齐)
    optical_y = np.cross(optical_z, optical_x)

    # 构建 optical frame 在世界中的旋转矩阵
    # 列分别是 optical X, Y, Z 在世界中的方向
    R_optical_in_world = np.column_stack([optical_x, optical_y, optical_z])

    # 从标定结果得知: optical frame 相对 gripper 的旋转
    # optical_X = ?, optical_Y = ?, optical_Z = ? (均以 gripper 轴表示)
    # R_optical_from_gripper: 3x3 矩阵，将 gripper 轴映射到 optical 轴
    #
    # 示例 (需根据 Task 3 结果替换):
    #   若 optical Z = -gripper Y, optical X = -gripper X, optical Y = gripper Z:
    #   R_optical_from_gripper = [[-1,0,0], [0,0,1], [0,-1,0]]
    #
    # 注: 此矩阵的列 = [optical_X_in_gripper, optical_Y_in_gripper, optical_Z_in_gripper]
    R_optical_from_gripper = np.array([
        [ 0,  0,  1],   # optical X = +gripper Z   ← 占位值, 等标定后替换
        [-1,  0,  0],   # optical Y = -gripper X
        [ 0, -1,  0],   # optical Z = -gripper Y
    ])

    # R_gripper_in_world = R_optical_in_world @ R_optical_from_gripper.T
    # 因为 optical = gripper @ R_optical_from_gripper (按列变换)
    # gripper = optical @ R_optical_from_gripper^T
    R_gripper = R_optical_in_world @ R_optical_from_gripper.T

    r = Rotation.from_matrix(R_gripper)
    q = r.as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
```

**关键**：`R_optical_from_gripper` 矩阵必须用 Task 3 标定结果填充。矩阵的三列分别是 optical X, Y, Z 在 gripper 坐标系中的单位向量。

- [ ] **Step 3: 对标定中收集的所有姿态验证数学自洽性**

用 Python 交互式验证：

```python
import numpy as np
from scipy.spatial.transform import Rotation

# 从标定数据中取 identity 姿态下的 optical 轴映射，填充 R_optical_from_gripper
# 例如: 若 identity 下 optical Z 指向 gripper Y 的负方向:
R_optical_from_gripper = np.array([[...], [...], [...]])

# 验证: 对 identity gripper 朝向 (R_gripper=I)
R_optical_predicted = np.eye(3) @ R_optical_from_gripper
# optical Z 列应该与标定中观察到的 identity optical Z 方向一致
print("Predicted optical Z in world:", R_optical_predicted[:, 2])
print("Observed optical Z in world: 从标定表填入")
```

- [ ] **Step 4: 提交**

```bash
git add src/position_topic/position_topic/camera_utils.py
git commit -m "fix(camera_utils): rewrite compute_lookat_quaternion with calibrated axis mapping"
```

---

### Task 5: 修正 PRE_GRASP 四元数

**Files:**
- Modify: `src/position_topic/position_topic/grab_action.py:342-352`

当前 PRE_GRASP 用 `Ry(-90°)` (`y=-sqrt2/2, w=sqrt2/2`)，注释写 "光轴 (gripper X) 指向下方"。但此映射未经标定确认——可能不是 gripper X，且方向的正负号可能反了。

- [ ] **Step 1: 确定夹爪指尖方向**

在 Gazebo 中让机械臂移到 PRE_GRASP 位姿（可在 Task 3 标定 launch 中最后一个姿态就是 PRE_GRASP），观察：
- gripper 的哪个轴 (红/绿/蓝) 指向下方？
- 哪个轴沿着指尖张开方向？

- [ ] **Step 2: 推算 PRE_GRASP 需要的四元数**

PRE_GRASP 要求：末端垂直向下（gripper approach direction = 世界 -Z）。

如果从标定确认 gripper 的 Z+ 指向下方（或 X+ 指向下方），用对应的 Rotation 生成四元数。

例如，若 gripper X+ 指向下方（当前猜测）：
```python
from scipy.spatial.transform import Rotation
# gripper X → world -Z, gripper Y → world +Y, gripper Z → world +X
# 这是 Rx(-90°) 的效果 (X 轴向自己转 -90°)
r = Rotation.from_euler('x', -90, degrees=True)
```

- [ ] **Step 3: 更新代码**

替换 `grab_action.py:348-352` 中的 `Quaternion` 为标定确认的正确值。

```python
# 修改前 (当前代码, 靠猜测):
# sqrt2_2 = math.sqrt(2) / 2.0
# pre_grasp_pose.orientation = Quaternion(
#     x=0.0, y=-sqrt2_2, z=0.0, w=sqrt2_2,
# )

# 修改后 (使用 scipy 生成, 根据标定结果调整):
from scipy.spatial.transform import Rotation
r = Rotation.from_euler('x', -90, degrees=True)  # 或其他轴, 取决于标定结果
q = r.as_quat()
pre_grasp_pose.orientation = Quaternion(
    x=q[0], y=q[1], z=q[2], w=q[3],
)
```

- [ ] **Step 4: 提交**

```bash
git add src/position_topic/position_topic/grab_action.py
git commit -m "fix(grab_action): correct PRE_GRASP orientation based on calibration"
```

---

### Task 6: 端到端集成测试

**目的**：确认修复后整个抓取流程能正常工作。

- [ ] **Step 1: 干跑 VLM 推理测试**

```bash
# 启动完整仿真 (无草莓/纯方块场景, 更快更稳定)
ros2 launch position_topic test_no_strawberry.launch.py
```

在另一个终端：

```bash
# 触发 VLM 阶段一
ros2 topic pub /task_command std_msgs/msg/String "data: '找出最适合抓取的目标物体'"

# 观察日志输出
ros2 topic echo /target_observation_pose --once
ros2 topic echo /grab_status
```

- [ ] **Step 2: 验证观察位姿正确性**

在 Gazebo 中观察：机械臂到达观察位姿后，ee_camera_optical_link 的蓝色 Z 轴是否正确指向目标物体。

- [ ] **Step 3: 如正常，切换到草莓场景测试**

```bash
ros2 launch position_topic move_demo.launch.py
```

- [ ] **Step 4: 记录测试结果**

在内存文件中更新状态：相机朝向是否已正确，抓取成功率如何。

---

## 备注

- **最不确定的部分**：`R_optical_from_gripper` 矩阵的具体值，完全依赖 Task 3 的标定结果。在标定完成之前，Task 4 中的矩阵值是占位的。
- **降级策略**：如果标定后发现 optical frame 和 gripper 之间的轴映射与当前 URDF RPY 根本上不兼容，可能需要直接修改 URDF 中的 `origin_rpy` 参数使映射关系更简单（例如让 optical Z 直接等于 gripper 的某个轴），但这属于计划外的调整。
- **相机安装角**：当前相机安装 RPY=`0 0 0`（无旋转），但相机本体可能不是垂直向下——可能需要一个小俯角（pitch）。如果 end-to-end 测试发现视角不佳，后续可微调 `origin_rpy`。
