# SO101 农业采摘机械臂 — 具身抓取项目

> ROS 2 Humble + Gazebo Classic + MoveIt 2 + VLM (Qwen3-VL-Plus via DashScope)

## 项目结构

```
lerobot_ws/src/
├── lerobot_description/    # SO101 URDF + Gazebo 仿真世界
├── lerobot_controller/     # ros2_control 控制器配置
├── lerobot_moveit/         # MoveIt 2 运动规划配置 + SRDF
├── position_topic/         # 核心抓取逻辑 ← 主要开发包
│   ├── vlm_bridge.py       # VLM 多目标推理 + rank_targets 排序 + 循环采摘控制
│   ├── grab_action.py      # 11 阶段抓取状态机 + PUSH_SWEEP 推扫排障 + MoveIt
│   ├── vlm_client.py       # DashScope OpenAI 兼容 API 调用封装 (max_tokens=500)
│   ├── camera_utils.py     # 深度查询、像素反投影、TF2 变换、LookAt 四元数
│   ├── prompts.py          # VLM System Prompt 模板 (阶段一多目标/阶段二)
│   ├── push_sweep.py       # 推扫排障轨迹生成 (3 Cartesian 航点)
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

# 触发阶段一 VLM 推理 (多目标识别)
ros2 topic pub --once /task_command std_msgs/msg/String "data: '找出所有可采摘的草莓果实'"
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | VLM API 密钥 (Qwen3-VL-Plus) |

## 抓取流程状态机 (11 阶段)

```
IDLE → SETUP → MOVE_TO_OBSERVE → AWAIT_STAGE2 → PRE_GRASP
  → [PUSH_SWEEP]  ← 仅当 obstacle_above != "none" 且 material == "soft"
  → APPROACH → GRASP → LIFT_PLAN → TRANSPORT_PLAN → PLACE → IDLE (重新等待)
```

| 状态 | 行为 |
|------|------|
| IDLE | 等待 `/target_observation_pose` + `/target_pre_grasp` 协同到达 |
| SETUP | 注册目标碰撞物体到 Planning Scene (于 P_rough)，重置失败计数 |
| MOVE_TO_OBSERVE | MoveIt 移动到斜上方观察位姿 (含方向约束, 放宽位置/姿态容差) |
| AWAIT_STAGE2 | 等待 VLM 阶段二精确定位 `/target_pose`，超时降级用 P_rough |
| PRE_GRASP | 移除 scene 目标 → MoveIt 移动到 P_precise 正上方 (Z+0.08m)，末端垂直向下 |
| PUSH_SWEEP | [可选] Cartesian 推扫软障碍物 (3 航点)，失败降级为直接 APPROACH |
| APPROACH | 张爪 → Cartesian 直线下探 (max_step=0.01, avoid_collisions=False) |
| GRASP | 闭合夹爪 → Gazebo AttachLink + MoveIt attach_object 双层附着 |
| LIFT_PLAN | 抬起至预抓取高度 |
| TRANSPORT_PLAN | 移动到放置位置 |
| PLACE | 双层脱离 + 张爪释放 → 发布 "grasp_complete" 触发循环 |

## Topic 通信

| Topic | 方向 | 说明 |
|-------|------|------|
| `/target_observation_pose` | vlm_bridge→grab_action | 阶段一观察位姿 (含 LookAt 四元数) |
| `/target_pre_grasp` | vlm_bridge→grab_action | 阶段一 P_rough (粗定位 3D，用于碰撞注册和降级) |
| `/target_metadata` | vlm_bridge→grab_action | 当前目标语义属性 (JSON: material, obstacle_above, direction, ripeness, label) |
| `/target_pose` | vlm_bridge→grab_action | 阶段二精确定位位姿 |
| `/grab_status` | grab_action→vlm_bridge | 状态反馈 ("stage1_done"/"grasp_complete"/"error") |
| `/task_status` | vlm_bridge→外部 | 任务完成信号 ("task_complete") |
| `/task_command` | 外部→vlm_bridge | 触发阶段一推理的用户指令 |

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pre_grasp_z_offset` | 0.08 | 预抓取点高于目标的高度 (m) |
| `jaw_clearance` | 0.03 | APPROACH 目标点 Z 向补偿 (m) |
| `stage2_timeout` | 3.0 | 阶段二等待超时 (s) |
| `fallback_observation_points` | [0.20,0.0,0.22, 0.18,0.0,0.20, 0.22,0.05,0.18] | 降级观察点候补 (平铺 list, 每 3 个一组 xyz) |
| `velocity_scaling` | 0.3 | 全局速度缩放 |
| `planning_time` | 5.0 | MoveIt 规划时间上限 (s) |
| `dry_run` (vlm_bridge) | False | 不发布消息，仅测试 VLM 推理 |

## 观察位姿安全机制

- **方向偏移**: 阶段一 VLM 返回 `optimal_approach_direction`，从 6 个方向选择观察点
- **X 钳制**: 观察点 X < 0.10m 强制重置 (SO101 基座死区防护)
- **容差放宽**: 观察阶段 pos_tolerance=0.02, ori_tolerance=0.2 (PRE_GRASP/APPROACH 恢复严苛)
- **动态观察点失败降级**: 移除朝向约束 (use_orientation=False, 因为 position_only_ik=True 时 KDL 无法同时满足朝向) → 依次尝试 `fallback_observation_points` 候补列表 (每 3 个一组 xyz) → 全部失败则转 FAILED
- **笛卡尔直线**: APPROACH 必须用 compute_cartesian_path，禁止关节空间插值

## 已知约定

- **VLM**: Qwen3-VL-Plus，bbox 使用归一化坐标 [0,1000]，非像素坐标；max_tokens=500
- **深度**: Gemini 335 输出 32FC1 (float, m)，ee_camera 输出 16UC1 (uint16, mm)
- **Prompts**: STAGE1 输出 `targets[]` + `obstacles[]` 数组；STAGE2 输出单目标精确定位
- **多目标排序**: rank_targets() 按 成熟度(ripe>overripe>unripe) > 无障碍(none>leaf>stem) > 置信度 > 索引
- **软硬分类**: VLM 标注 material ("soft"/"hard")；soft 可抓可推扫，hard 必须绕行
- **推扫触发**: obstacle_above != "none" 且 material == "soft" → PUSH_SWEEP 状态
- **循环控制**: grasp_complete → 重新 Stage1；连续 2 次空 targets → task_complete；同目标 3 次失败 → 黑名单
- **光学帧 RPY**: `(π, 0, -π/2)` → `optical X=-gripper Y, optical Y=-gripper X, optical Z=-gripper Z`
  - 旋转矩阵: R = Rz(-π/2)·Rx(π) = `[[0,-1,0],[-1,0,0],[0,0,-1]]`
  - `compute_lookat_quaternion`: `gripper Z = -direction`，无 ad-hoc 补偿
- **相机 frame_name**: `ee_camera_optical_link` (已修复，之前 Bug 是 `camera_optical_link`)

## 已知未解决问题

- **相机朝向**: ✅ 已解决 (2026-06-12)。关键发现: Gazebo depth camera 渲染轴为 optical X (红)，非 optical Z (蓝)。`compute_lookat_quaternion` 已修正为 optical X 对准目标。
- **vlm_bridge error 分支死代码**: ✅ 已修复 (2026-06-13)。重复 `elif msg.data == "error"` 分支导致 `_handle_grab_error` (黑名单+失败计数+换目标) 永远不可达。已删除第一个冗余分支。
- **观察位姿 IK**: 部分修复 (2026-06-13)。降级观察点已移除朝向约束 (use_orientation=False) 并改为多候补参数化列表 `fallback_observation_points`。但动态观察点仍保留朝向约束，position_only_ik=True 下 KDL 无法满足，仍可能 IK 失败→触发降级链。
- **相机位置**: X+ 和 Y/Z 轴偏移效果未全部测试完成。
- **PRE_GRASP 朝向**: `Ry(-90°)` 使 gripper X (approach direction) 指向世界 -Z。待实测验证指尖方向是否与夹爪物理匹配。
- **AttachLink 脱离后物体翻转**: DetachLink 后 target_box 在 Gazebo 中周期性翻转 180°。
- **_consecutive_failures 计数器未消费**: grab_action 中失败计数递增但无人检查——FAILED→IDLE 立即转换，不做重试上限。vlm_bridge 侧黑名单逻辑依赖此计数但因 error 分支 bug 从未生效 (已修复 blacklist 侧，但 grab_action 侧仍无自主中断)。

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

# 查看 VLM 多目标识别结果
ros2 topic echo /target_metadata --once

# 查看任务完成状态
ros2 topic echo /task_status

# dry_run 模式测试 VLM (不控制机械臂)
ros2 run position_topic vlm_bridge --ros-args -p dry_run:=true
```
