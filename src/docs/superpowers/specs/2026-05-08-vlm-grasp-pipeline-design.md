# VLM + ROS 2 抓取管线设计规范

> 日期：2026-05-08  
> 范围：第一阶段——单目标无遮挡抓取闭环

## 1. 架构概览

### 1.1 组件与数据流

```
用户 /task_command (std_msgs/String)
         │
         ▼
   ┌─────────────┐
   │ vlm_bridge  │  阶段一: Gemini335 全局推理 → 粗定位
   │             │  阶段二: ee_camera 近距精定位
   │             │  VLM: Qwen3 VL Plus (DashScope OpenAI 兼容接口)
   │             │  发布: /target_pre_grasp, /target_pose
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ grab_action │  订阅: /target_pre_grasp, /target_pose
   │             │  MoveGroup: arm 规划组
   │             │  Service: /ATTACHLINK, /DETACHLINK
   │             │  发布: /grab_status (状态反馈)
   └─────────────┘
```

### 1.2 双相机分工

| 阶段 | 相机 | 用途 | 分辨率 | 深度范围 |
|------|------|------|--------|----------|
| 一 | Gemini 335（外部俯视） | 全局场景理解、目标发现、粗定位 | 640×480@30Hz | 0.25-2.5m |
| 二 | ee_camera（手眼） | 近距抓取点精确确认 | 640×480@30Hz | 0.05-8.0m |

### 1.3 调用链路（单次抓取）

1. 用户发布 `/task_command`
2. vlm_bridge 用 Gemini335 当前帧调用 VLM，解析 JSON，反投影得到 3D 粗定位
3. vlm_bridge 发布 `/target_pre_grasp`
4. grab_action MoveIt 规划 → 机械臂到达 pre-grasp 位姿
5. grab_action 发布 `/grab_status` = `"stage1_done"`
6. vlm_bridge 用 ee_camera 当前帧再次调用 VLM，解析 JSON，反投影得到 3D 精定位
7. vlm_bridge 发布 `/target_pose`
8. grab_action Cartesian 下移 → 夹取(AttachLink) → 抬起 → 搬运 → 放置(DetachLink)

---

## 2. vlm_bridge 详细设计

### 2.1 接口

**输入订阅：**

| Topic | 类型 | 用途 |
|-------|------|------|
| `/task_command` | `std_msgs/String` | 用户自然语言指令，触发推理 |
| `/camera/color/image_raw` | `sensor_msgs/Image` | Gemini335 RGB |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Gemini335 深度 |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Gemini335 内参 |
| `/so101/camera/image_raw` | `sensor_msgs/Image` | ee_camera RGB |
| `/so101/camera/depth/image_raw` | `sensor_msgs/Image` | ee_camera 深度 |
| `/so101/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | ee_camera 内参 |
| `/grab_status` | `std_msgs/String` | 抓取状态反馈（来自 grab_action） |

**输出发布：**

| Topic | 类型 | 用途 |
|-------|------|------|
| `/target_pre_grasp` | `geometry_msgs/PoseStamped` | 阶段一粗定位，frame_id="base" |
| `/target_pose` | `geometry_msgs/PoseStamped` | 阶段二精定位，frame_id="base" |

### 2.2 状态机

```
IDLE ──收到/task_command + Gemini335帧已就绪──→ STAGE1_QUERY
                                                    │
                                                    │ 编码RGB → b64
                                                    │ 调用 VLM(全局提示词)
                                                    │ 解析 JSON → 查深度 → 反投影 → TF2(camera_depth_optical→base)
                                                    │ 发布 /target_pre_grasp
                                                    │
                                                    ▼
                                              STAGE1_WAIT ──收到/grab_status="stage1_done" + ee_camera帧已就绪──→ STAGE2_QUERY
                                                                                                                    │
                                                                                                                    │ 编码RGB → b64
                                                                                                                    │ 调用 VLM(精定位提示词)
                                                                                                                    │ 解析 JSON → 查深度 → 反投影 → TF2(ee_camera_optical→base)
                                                                                                                    │ 发布 /target_pose
                                                                                                                    │
                                                                                                                    ▼
                                                                                                                  IDLE
```

### 2.3 VLM 调用

- API: DashScope OpenAI 兼容接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 模型: `qwen3-vl-plus`
- 输入: 用户自然语言指令 + 单帧 base64 JPEG
- 输出: JSON（单目标无遮挡格式: `{"target": {"pixel_x": int, "pixel_y": int, "label": str, "confidence": float}}`）
- API Key: 从环境变量 `DASHSCOPE_API_KEY` 读取

### 2.4 2D→3D 映射

```
1. 解析 VLM JSON → (u, v)
2. 从 depth_msg 取深度: Z = depth_msg.data[v * width + u] / 1000.0 (m)
   无效深度 → 3×3 邻域中值回退
3. 反投影 (CameraInfo.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]):
   X_cam = (u - cx) * Z / fx
   Y_cam = (v - cy) * Z / fy
   Z_cam = Z
4. TF2 变换:
   阶段一: camera_depth_optical_frame → base
   阶段二: ee_camera_optical_link → base
```

### 2.5 文件结构

```
position_topic/position_topic/
├── vlm_bridge.py      ← 主节点（状态机、ROS 接口）
├── vlm_client.py      ← VLM API 调用封装（编码、发送、解析）
├── camera_utils.py    ← 像素反投影、深度查询、TF2 变换
└── prompts.py         ← 阶段一/阶段二 System Prompt 字符串常量
```

---

## 3. grab_action 详细设计

### 3.1 接口

**输入订阅：**

| Topic | 类型 | 用途 |
|-------|------|------|
| `/target_pre_grasp` | `geometry_msgs/PoseStamped` | 预抓取位姿（上方 waypoint） |
| `/target_pose` | `geometry_msgs/PoseStamped` | 精确抓取位姿 |

**输出发布：**

| Topic | 类型 | 用途 |
|-------|------|------|
| `/grab_status` | `std_msgs/String` | 状态反馈: `"stage1_done"` / `"stage2_done"` / `"grasp_complete"` / `"error"` |

**调用：**
| 服务 | 类型 | 用途 |
|------|------|------|
| `/move_action` | `MoveGroup` | MoveIt 2 arm 规划组 |
| `/ATTACHLINK` | `AttachLink` | 附着物体到 gripper |
| `/DETACHLINK` | `DetachLink` | 释放物体 |

### 3.2 状态机

```
IDLE ←────────────────────────────────────────────────（回到起点）
  │                                                     ↑
  │ 收到 /target_pre_grasp                              │
  ▼                                                     │
PRE_GRASP_PLAN ──失败──→ FAILED                         │
  │ 成功                                               │
  ▼                                                     │
PRE_GRASP_EXEC ──失败──→ FAILED                         │
  │ 成功                                               │
  │ 发布 "stage1_done"                                  │
  ▼                                                     │
  │ 等待 /target_pose（超时10s → Z偏移降级）             │
  ▼                                                     │
APPROACH_PLAN ──失败──→ FAILED                          │
  │ Cartesian 直线下移（步长 1cm）                      │
  ▼                                                     │
APPROACH_EXEC                                           │
  │                                                     │
  ▼                                                     │
GRASP                                                   │
  │ 闭合夹爪(关节6→目标值)                               │
  │ 调用 /ATTACHLINK                                    │
  ▼                                                     │
LIFT_PLAN                                               │
  │ Cartesian 反向抬起(步长 1cm)                        │
  ▼                                                     │
LIFT_EXEC                                               │
  │                                                     │
  ▼                                                     │
TRANSPORT_PLAN                                          │
  │ 自由规划到放置位姿                                   │
  ▼                                                     │
TRANSPORT_EXEC                                          │
  │                                                     │
  ▼                                                     │
PLACE                                                   │
  │ 调用 /DETACHLINK                                    │
  │ 张开夹爪                                            │
  │ 发布 "grasp_complete" ────────────────────────────────┘
  └──失败→ FAILED
```

### 3.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `placement_position` | `[0.2, -0.15, 0.25]` | 放置位姿（base 系），硬编码 |
| `pre_grasp_z_offset` | `0.08` | pre-grasp 在目标上方的高度(m) |
| `approach_step` | `0.01` | Cartesian 路径步长(m) |
| `gripper_close_pos` | `0.5` | 夹爪闭合位置(关节6归一化值) |
| `gripper_open_pos` | `0.0` | 夹爪张开位置(关节6归一化值) |
| `stage2_timeout` | `10.0` | 等待精定位结果的超时(s) |
| `planning_time` | `5.0` | MoveIt 允许规划时间(s) |
| `velocity_scaling` | `0.3` | 速度缩放 |

### 3.4 降级策略

阶段二超时未收到 `/target_pose` 时，不放弃抓取——直接用 `/target_pre_grasp` 的 Z 减去 `pre_grasp_z_offset` 作为抓取位姿，跳过精定位。阶段一 Gemini335 的粗定位虽精度有限，但在无遮挡场景下通常能满足基本抓取。

### 3.5 文件

```
position_topic/position_topic/
└── grab_action.py      ← 主节点（状态机 + MoveIt + LinkAttacher 调用）
```

---

## 4. 与现有代码的关系

| 文件 | 操作 | 说明 |
|------|------|------|
| `position_topic/vlm_bridge.py` | **重写** | 当前为空壳 |
| `position_topic/grab_action.py` | **新建** | 抓取流程状态机 |
| `position_topic/camera_utils.py` | **新建** | 反投影 + TF 工具函数 |
| `position_topic/prompts.py` | **新建** | VLM 提示词模板 |
| `position_topic/setup.py` | **修改** | 注册新 entry_points |
| `position_topic/launch/move_demo.launch.py` | **修改** | 添加 vlm_bridge + grab_action 节点 |
| `position_topic/position_subscriber.py` | **不动** | 保留用于手动调试 |
| `position_topic/position_publisher.py` | **不动** | 保留用于手动调试 |

---

## 5. 测试策略

### 5.1 单元级测试

- `camera_utils.py`: 给定已知 u/v/Z/K，验证反投影结果正确
- `vlm_client.py`: Mock VLM 返回固定 JSON，验证解析逻辑
- `grab_action.py` 状态机: 手动发布不同状态触发词，检查状态转换

### 5.2 集成测试（Gazebo 中验证）

- 单 target_box 无遮挡，硬编码 `/target_pre_grasp`，验证抓取全流程
- 单 target_box + VLM 两阶段，验证端到端
- 单草莓植株 + VLM，验证自然目标

### 5.3 手动测试便利性

- `position_publisher.py` + `position_subscriber.py` 保留，可随时跳过大模型独立测试 MoveIt 通路
- vlm_bridge 支持 `--ros-args -p dry_run:=true` 模式：只推理不发位姿，方便调 prompt
