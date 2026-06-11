# 多目标软硬识别与顺序采摘 设计文档

> **日期:** 2026-06-12
> **状态:** 设计确认，进入实现计划阶段

## 目标

将当前"单次抓取最优目标"的 VLM 抓取 pipeline 升级为"识别所有目标 → 属性分类 → 优先级排序 → 排障 → 顺序逐个采摘"的完整农业采摘流程。

## 背景

当前系统已完成 VLM 双阶段单目标抓取：Gemini 335 俯视全局推理 → ee_camera 近距精定位 → 10 阶段抓取状态机。但每次只处理一个目标（VLM 选"最适合的"），抓完即结束。

农业采摘需要：识别场景中所有可抓果实、区分软硬（果实 vs 茎杆）、判断成熟度、处理叶片遮挡、循环采摘直到清空。

## 架构

```
当前流程:
  Stage1 VLM → 单个目标 → grab → place → END

新流程:
  Stage1 VLM → 目标列表(含属性) + 障碍物列表
    → 代码层优先级排序 → 选最优目标
    → 障碍物注册到 Planning Scene
    → [可选] 推扫排障
    → 观察 → Stage2 → 抓取 → 放置
    → 重新 Stage1 → 直到无目标 → END
```

### 改动文件清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `prompts.py` | **改** | STAGE1 prompt 输出目标列表 + 属性字段（material/ripeness/obstacle_above）；STAGE2 prompt 增加障碍物标注 |
| `vlm_bridge.py` | **改** | 阶段一解析多目标列表；新增优先级排序函数；PLACE 后自动重新触发阶段一；循环终止判断 |
| `grab_action.py` | **改** | 新增 PUSH_SWEEP 状态（排障推扫）；障碍物碰撞注册；PLACE 后回到 IDLE 而非结束 |
| `camera_utils.py` | 不改 | 现有接口够用 |
| `planning_scene_manager.py` | 不改 | add_object/remove_object 接口已支持动态管理 |
| **新** `push_sweep.py` | **新** | 推扫轨迹生成模块：水平弧线推扫 + 返回原位置 |
| `move_demo.launch.py` | 微调 | 可能需要加入循环行为相关参数 |

### 节点职责

- **vlm_bridge**: 负责所有"智能"——VLM 调用、结果解析、优先级排序、循环控制、目标/观察位姿发布
- **grab_action**: 负责所有"运动"——状态机执行、Planning Scene 管理、夹爪控制、推扫轨迹执行
- **push_sweep**: 纯工具模块，给定起点终点生成推扫轨迹，被 grab_action 调用

## 关键设计

### 1. VLM Stage1 输出结构（单次调用，Prompt 增强）

VLM 一次调用同时输出目标列表和障碍物列表。不拆成两次调用——Qwen3-VL-Plus 完全有能力在单次推理中输出这两个列表，拆分会倍增加延迟和成本。

```json
{
  "targets": [
    {
      "bbox": [300, 200, 450, 380],
      "label": "strawberry",
      "confidence": 0.92,
      "material": "soft",
      "ripeness": "ripe",
      "obstacle_above": "none",
      "optimal_approach_direction": "top"
    },
    {
      "bbox": [600, 400, 750, 550],
      "label": "strawberry",
      "confidence": 0.85,
      "material": "soft",
      "ripeness": "unripe",
      "obstacle_above": "leaf",
      "optimal_approach_direction": "left"
    }
  ],
  "obstacles": [
    {"bbox": [500, 100, 700, 300], "label": "stem", "material": "hard"},
    {"bbox": [200, 150, 400, 350], "label": "leaf", "material": "soft"}
  ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `targets[].bbox` | [4]int | 归一化 [0,1000] 边界框 |
| `targets[].label` | str | 物体描述，如 "strawberry" |
| `targets[].confidence` | float | 置信度 0-1 |
| `targets[].material` | "soft" \| "hard" | 软硬分类——决定抓取力度和排障策略 |
| `targets[].ripeness` | "ripe" \| "unripe" \| "overripe" | 成熟度——决定采摘优先级 |
| `targets[].obstacle_above` | str | 上方障碍物类型：`"none"` / `"leaf"` / `"stem"` |
| `targets[].optimal_approach_direction` | str | 枚举值不变：`front`/`left`/`right`/`front_left`/`front_right`/`top` |
| `obstacles[].material` | "soft" \| "hard" | 软障碍物可推扫，硬障碍物必须绕行 |

### 2. 优先级排序（代码层，非 VLM）

排序逻辑放在 `vlm_bridge.py` 中，不依赖 VLM 输出排序号。排序 key：

```python
# 排序优先级: 成熟度 > 无障碍 > 置信度 > 索引
RIPENESS_ORDER = {"ripe": 0, "overripe": 1, "unripe": 2}
OBSTACLE_ORDER = {"none": 0, "leaf": 1, "stem": 2}

def rank_targets(targets):
    def sort_key(t, idx):
        ripeness_score = RIPENESS_ORDER.get(t.get("ripeness", "unripe"), 2)
        obstacle_score = OBSTACLE_ORDER.get(t.get("obstacle_above", "leaf"), 2)
        return (ripeness_score, obstacle_score, -t.get("confidence", 0), idx)
    return sorted(targets, key=lambda t: sort_key(t[1], t[0]) if isinstance(t, tuple) else sort_key(t, 0))
```

**规则变化（未来可扩展）：** 如果没有 `ripe` 目标但有 `overripe` 目标，仍然采摘 overripe（避免腐烂），但优先 ripe。

### 3. 推扫排障

简单轨迹——末尾执行器从目标侧上方水平平移一小段距离，模拟"用手拨开叶子"。

**触发条件：** `obstacle_above != "none"` 且 `obstacle.material == "soft"`（硬障碍物不推扫，改为调整 approach 方向绕行）。

**轨迹生成：**

```
推扫起点: P_target + (0, 0, 0.05)          # 目标正上方 5cm
推扫中点: P_target + (0, sweep_offset, 0.05) # 水平推 8cm（方向由 VLM optimal_approach_direction 决定）
推扫终点: 回到起点                            # 回原位后正常下探

SWEEP_DIRECTION_OFFSETS = {
    "front":  ( 0.00,  0.00),  # front 不推，直接下探
    "left":   ( 0.00,  0.08),  # 从左接近 → 向右推
    "right":  ( 0.00, -0.08),  # 从右接近 → 向左推
    "front_left":  ( 0.00,  0.06),
    "front_right": ( 0.00, -0.06),
    "top":    ( 0.00,  0.00),  # 顶部接近不推
}
```

**执行方式：** Cartesian 轨迹，3 个航点（起点→推扫点→起点），`avoid_collisions=False`。

### 4. 障碍物碰撞注册

VLM 阶段一输出的 `obstacles` 列表，选取置信度较高的障碍物（confidence > 阈值），将其注册到 Planning Scene 作为碰撞物体：

- **软障碍物（leaf）：** 注册为碰撞物体，推扫前临时移除，推扫后不再重新注册（已被推开）
- **硬障碍物（stem）：** 始终注册为碰撞物体，MoveIt 自动绕行

障碍物物体用长方体/球体近似（根据 bbox 深度估算 3D 尺寸）。

### 5. 循环控制

```
vlm_bridge: Stage1 → 发布列表 → grab_action 抓取一个目标
grab_action: PLACE → 发布 "grasp_complete"
vlm_bridge: 收到 "grasp_complete" → 重新 Stage1
  如果 VLM 返回空 targets 列表 → 发布 "task_complete" → END
  如果 targets 非空 → 选最优 → 发布新目标 → 循环
```

**终止条件：**
1. VLM 返回空 `targets` 列表（无果实可见）
2. 所有 targets 置信度低于阈值（默认 0.3，可通过参数调整）
3. 连续 3 次抓取失败（grab_action 返回 "error"）
4. 用户手动终止

### 6. Grab Action 状态机变更

新增状态 `PUSH_SWEEP`，插入在 PRE_GRASP 和 APPROACH 之间：

```
SETUP → MOVE_TO_OBSERVE → AWAIT_STAGE2 → PRE_GRASP
  → [PUSH_SWEEP]  ← 仅当 obstacle_above != "none" 且 material == "soft"
  → APPROACH → GRASP → LIFT_PLAN → TRANSPORT_PLAN → PLACE → IDLE (重新等待)
```

**PUSH_SWEEP 状态行为：**
1. 调用 `push_sweep.generate_sweep_trajectory(pre_grasp_pose, direction)` 生成推扫航点
2. 通过 arm_controller 的 `compute_cartesian_path` 服务规划 Cartesian 轨迹
3. 执行轨迹
4. 成功后跳转 APPROACH

### 7. 向后兼容

- `test_no_strawberry.launch.py` 的方块场景兼容：方块 `material` 通常为 `"hard"`，不会触发推扫
- `grab_action` 的 PUSH_SWEEP 状态为可选——如果目标 `obstacle_above == "none"`，直接跳过进入 APPROACH
- 已有参数（`pre_grasp_z_offset`、`jaw_clearance` 等）不变

### 8. 障碍物通信接口

grab_action 需要知道当前目标的 `obstacle_above` 和 `material` 属性才能决定是否推扫，但 `PoseStamped` 消息无法承载这些语义字段。方式：

**方案：扩展 `/target_pre_grasp` 的语义，同时发布障碍物元信息**

vlm_bridge 在发布 `/target_pre_grasp` 的同时，将当前选中的目标属性（`material`、`obstacle_above`、`optimal_approach_direction`）通过一个新的 String topic `/target_metadata` 以 JSON 格式发布。grab_action 订阅此 topic 解析元信息。

```
/target_metadata (std_msgs/String, JSON):
{
  "material": "soft",
  "obstacle_above": "leaf",
  "optimal_approach_direction": "left",
  "ripeness": "ripe",
  "label": "strawberry"
}
```

**障碍物碰撞注册**：vlm_bridge 不再单独发布障碍物。障碍物 3D 位置由 grab_action 在 SETUP 阶段通过 `/target_metadata` 中的 `obstacle_above` 信息决定是否注册（硬障碍物注册为碰撞物体）。推扫排障由 grab_action 在 PUSH_SWEEP 状态自主执行，vlm_bridge 不参与。

简化理由：软障碍物通过推扫处理，硬障碍物在当前 SO101 工作空间内极少出现（茎杆通常不在抓取路径上）。如果 VLM 报告 `obstacle_above == "stem"`（硬），grab_action 降级为跳过该目标。

## 错误处理

| 场景 | 处理 |
|------|------|
| VLM 返回空 targets + 非空 obstacles | 发布 warning 日志，障碍物注册到 scene 后重新 Stage1（可能障碍物挡住了视线） |
| VLM 连续 2 次返回空 targets | 终止循环，发布 "task_complete" |
| 推扫轨迹规划失败 | 降级为直接 APPROACH（不推扫），日志 warning |
| 推扫执行失败 | 同上，直接 APPROACH |
| 某个目标抓取失败 3 次 | 跳过该目标（加入黑名单），选下一个 |
| Stage2 VLM 超时 | 用降级 P_rough（现有逻辑不变） |

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│ vlm_bridge                                                   │
│                                                              │
│   /task_command ──→ Stage1 VLM ──→ parse targets[]          │
│                        │              obstacles[]            │
│                        ↓                                     │
│                  rank_targets()                              │
│                        │                                     │
│                  选最优 target                                │
│                        │                                     │
│          ┌─────────────┼──────────────┐                      │
│          ↓             ↓              ↓                      │
│   /target_pre_grasp  /target_observation_pose  /target_metadata   │
│                                                              │
│   Stage2 VLM ← /grab_status="stage1_done"                    │
│        │                                                     │
│        ↓                                                     │
│   /target_pose                                               │
│                                                              │
│   /grab_status="grasp_complete" ──→ loop back to Stage1      │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────┐
│ grab_action            ↓                                     │
│   (订阅 /target_metadata 获取当前目标属性)                       │
│                                                              │
│   SETUP ──→ MOVE_TO_OBSERVE ──→ AWAIT_STAGE2                 │
│      │                          │                            │
│      │  注册 target collision    │                            │
│      │  object                  │                            │
│      └──────────────────────────┘                            │
│                │                                              │
│                ↓                                              │
│   PRE_GRASP ──→ [PUSH_SWEEP] ──→ APPROACH                    │
│                    (optional)                                 │
│      ↓                                              │
│   GRASP → LIFT → TRANSPORT → PLACE                            │
│      ↓                                                        │
│   /grab_status="grasp_complete" → IDLE                        │
└──────────────────────────────────────────────────────────────┘
```

## 非功能需求

- **延迟**: 推扫排障 ≤ 5s（3 个 Cartesian 航点，不含规划时间）
- **VLM 成本**: 每次 Stage1 输出目标数上限 10 个（token 控制），Max token 放宽到 500
- **可调试**: 每个阶段的 VLM 输出和排序结果通过 ROS 日志打印
- **可复用**: `push_sweep.py` 独立于抓取流程，可被其他节点调用

## 不会做的事（显式排除）

- 不做力控/力矩反馈推扫（Gazebo 仿真不支持）
- 不做视觉伺服（visual servoing）——保持当前 bbox + 深度反投影方案
- 不做多臂协作
- 不做果实成熟度视觉特征检测（靠 VLM 语义判断，不引入传统 CV 颜色阈值）
- 不做推扫力度自适应（固定 8cm 水平偏移量，通过参数可配）
