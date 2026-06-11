# SO101 农业采摘机械臂 — 具身抓取项目

> ROS 2 Humble + Gazebo Classic + MoveIt 2 + VLM (Qwen3-VL-Plus via DashScope)

## 项目结构

```
lerobot_ws/src/
├── lerobot_description/    # SO101 URDF + Gazebo 仿真世界
├── lerobot_controller/     # ros2_control 控制器配置
├── lerobot_moveit/         # MoveIt 2 运动规划配置 + SRDF
├── position_topic/         # 核心抓取逻辑 ← 主要开发包
│   ├── vlm_bridge.py       # 双阶段 VLM 推理 + 2D→3D 映射 + 目标发布
│   ├── grab_action.py      # 10 阶段抓取状态机 + MoveIt + LinkAttacher
│   ├── vlm_client.py       # DashScope OpenAI 兼容 API 调用封装
│   ├── camera_utils.py     # 深度查询、像素反投影、TF2 变换、LookAt 四元数
│   ├── prompts.py          # VLM System Prompt 模板
│   ├── planning_scene_manager.py  # Planning Scene 物体/碰撞/附着管理
│   ├── position_subscriber.py     # 调试用 — 手动目标位姿→MoveGroup
│   └── position_publisher.py      # 调试用 — 手动发布目标位姿
└── IFRA_LinkAttacher/      # Gazebo 物体附着/脱离插件
```

## 启动方式

```bash
# 构建
cd ~/lerobot_ws
colcon build --symlink-install --packages-select position_topic lerobot_description lerobot_moveit lerobot_controller
source install/setup.bash

# 启动仿真 + 控制器 + MoveIt + VLM + 抓取 (含草莓场景)
ros2 launch position_topic move_demo.launch.py

# 启动仿真 + 控制器 + MoveIt + VLM + 抓取 (无草莓/纯方块)
ros2 launch position_topic test_no_strawberry.launch.py

# 触发阶段一 VLM 推理
ros2 topic pub /task_command std_msgs/msg/String "data: '找出最适合抓取的目标物体'"
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | VLM API 密钥 (Qwen3-VL-Plus) |

## 抓取流程状态机 (10 阶段)

```
IDLE → SETUP → MOVE_TO_OBSERVE → AWAIT_STAGE2 → PRE_GRASP → APPROACH
  → GRASP → LIFT_PLAN → TRANSPORT_PLAN → PLACE → IDLE
```

| 状态 | 行为 |
|------|------|
| IDLE | 等待 `/target_observation_pose` + `/target_pre_grasp` 协同到达 |
| SETUP | 注册目标碰撞物体到 Planning Scene (于 P_rough) |
| MOVE_TO_OBSERVE | MoveIt 移动到斜上方观察位姿 (含方向约束, 放宽位置/姿态容差) |
| AWAIT_STAGE2 | 等待 VLM 阶段二精确定位 `/target_pose`，超时降级用 P_rough |
| PRE_GRASP | 移除 scene 目标 → MoveIt 移动到 P_precise 正上方 (Z+0.08m)，末端垂直向下 |
| APPROACH | 张爪 → Cartesian 直线下探 (max_step=0.01, avoid_collisions=False) |
| GRASP | 闭合夹爪 → Gazebo AttachLink + MoveIt attach_object 双层附着 |
| LIFT_PLAN | 抬起至预抓取高度 |
| TRANSPORT_PLAN | 移动到放置位置 |
| PLACE | 双层脱离 + 张爪释放 |

## Topic 通信

| Topic | 方向 | 说明 |
|-------|------|------|
| `/target_observation_pose` | vlm_bridge→grab_action | 阶段一观察位姿 (含 LookAt 四元数) |
| `/target_pre_grasp` | vlm_bridge→grab_action | 阶段一 P_rough (粗定位 3D，用于碰撞注册和降级) |
| `/target_pose` | vlm_bridge→grab_action | 阶段二精确定位位姿 |
| `/grab_status` | grab_action→vlm_bridge | 状态反馈 ("stage1_done" 触发阶段二) |
| `/task_command` | 外部→vlm_bridge | 触发阶段一推理的用户指令 |

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pre_grasp_z_offset` | 0.08 | 预抓取点高于目标的高度 (m) |
| `jaw_clearance` | 0.03 | APPROACH 目标点 Z 向补偿 (m) |
| `stage2_timeout` | 3.0 | 阶段二等待超时 (s) |
| `velocity_scaling` | 0.3 | 全局速度缩放 |
| `planning_time` | 5.0 | MoveIt 规划时间上限 (s) |
| `dry_run` (vlm_bridge) | False | 不发布消息，仅测试 VLM 推理 |

## 观察位姿安全机制

- **方向偏移**: 阶段一 VLM 返回 `optimal_approach_direction`，从 6 个方向选择观察点
- **X 钳制**: 观察点 X < 0.10m 强制重置 (SO101 基座死区防护)
- **容差放宽**: 观察阶段 pos_tolerance=0.02, ori_tolerance=0.2 (PRE_GRASP/APPROACH 恢复严苛)
- **降级**: 观察点规划失败时回退到硬编码安全点 [0.20, 0.0, 0.25]，仍正常触发阶段二
- **笛卡尔直线**: APPROACH 必须用 compute_cartesian_path，禁止关节空间插值

## 已知约定

- **VLM**: Qwen3-VL-Plus，bbox 使用归一化坐标 [0,1000]，非像素坐标
- **深度**: Gemini 335 输出 32FC1 (float, m)，ee_camera 输出 16UC1 (uint16, mm)
- **Prompts**: 已通用化为"目标物体"，兼容方块测试和草莓场景
- **光学帧 RPY**: `(π, 0, -π/2)` → `optical X=-gripper Y, optical Y=-gripper X, optical Z=-gripper Z`
  - 旋转矩阵: R = Rz(-π/2)·Rx(π) = `[[0,-1,0],[-1,0,0],[0,0,-1]]`
  - `compute_lookat_quaternion`: `gripper Z = -direction`，无 ad-hoc 补偿
- **相机 frame_name**: `ee_camera_optical_link` (已修复，之前 Bug 是 `camera_optical_link`)

## 已知未解决问题

- **相机朝向**: ✅ 已解决 (2026-06-12)。关键发现: Gazebo depth camera 渲染轴为 optical X (红)，非 optical Z (蓝)。`compute_lookat_quaternion` 已修正为 optical X 对准目标。
- **相机位置**: X+ 和 Y/Z 轴偏移效果未全部测试完成。
- **PRE_GRASP 朝向**: `Ry(-90°)` 使 gripper X (approach direction) 指向世界 -Z。待实测验证指尖方向是否与夹爪物理匹配。

## 调试常用命令

```bash
# 检查 TF 树
ros2 run tf2_tools view_frames

# 查看相机话题
ros2 topic echo /target_observation_pose --once
ros2 topic echo /grab_status

# 手动发布目标位姿调试 grab_action
ros2 topic pub /target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'base'}, pose: {position: {x: 0.2, y: 0.0, z: 0.1}, orientation: {w: 1.0}}}"

# 相机轴标定 (诊断工具)
ros2 launch position_topic camera_calibrate.launch.py
```
