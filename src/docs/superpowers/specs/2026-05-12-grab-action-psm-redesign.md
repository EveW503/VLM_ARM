# Grab Action Planning Scene 改造设计

> 日期：2026-05-12  
> 范围：用 moveit_msgs Planning Scene 服务重写抓取管线，解决"撞飞物体"问题，为 Phase 2 软硬排障铺路  
> 相关阅读：MoveIt MTC pick-and-place 教程、MoveIt Python API 教程、Planning Around Objects 教程

## 1. 背景与动机

### 1.1 当前问题

现有 `grab_action.py` 通过 `MoveGroup` action 规划抓取轨迹，存在三个根本缺陷：

1. **Planning Scene 空白**：目标物体（target_box/草莓）只在 Gazebo 物理世界中存在，MoveIt 完全不知道。规划的路径穿过物体中心，Gazebo 物理碰撞把物体弹飞。
2. **双层不一致**：LinkAttacher 在 Gazebo 层附着物体，但 MoveIt Planning Scene 不知道物体已附着——抬起搬运时不把物体当机器人一部分，路径可能再次碰撞。
3. **Phase 2 无法扩展**：软硬排障需要对 Planning Scene 做 `apply_collision_object`、`allow_collisions`、`attach_object` 等操作，当前架构无此能力。

### 1.2 环境约束

- ROS 2 Humble：**不支持 `moveit_py`**（Iron/Rolling only），**不支持 `moveit_commander`**
- 可用接口：`moveit_msgs.srv.ApplyPlanningScene`、`moveit_msgs.srv.GetPlanningScene`（Humble 标准服务，无需额外安装）

## 2. 架构概览

### 2.1 新组件

```
grab_action.py                       planning_scene_manager.py
┌────────────────────┐              ┌──────────────────────────┐
│ 状态机 (8 阶段)     │  调用        │ ApplyPlanningScene client │
│                    │────────────→│ GetPlanningScene client  │
│ MoveGroup action   │              │                          │
│ LinkAttacher       │              │ 封装: add/remove/attach/ │
│ Gripper action     │              │ detach/allow_forbid/     │
└────────────────────┘              │ get_current_scene        │
                                    └──────────────────────────┘
                                            │
                                   ApplyPlanningScene.srv
                                            │
                                            ▼
                                    MoveGroup (move_group 节点)
```

### 2.2 与 vlm_bridge 的关系

- `vlm_bridge.py` **不变**，继续发布 `/target_pre_grasp`、`/target_pose`
- `planning_scene_manager.py` 将来 Phase 2 也可被 vlm_bridge 调用来注册硬/软障碍物

## 3. planning_scene_manager.py 接口

**文件**: `position_topic/position_topic/planning_scene_manager.py`

```python
class PlanningSceneManager:
    def __init__(self, node: Node):
        # 内部持有:
        # - apply_client (ApplyPlanningScene)
        # - get_client (GetPlanningScene)

    # ── 物体管理 ──
    def add_object(self, object_id, shape_type, dimensions, pose, frame_id)
        -> bool
    def remove_object(self, object_id) -> bool
    def remove_all_objects() -> bool

    # ── 碰撞策略 ──
    def allow_collision(self, link1, object_id) -> bool
    def forbid_collision(self, link1, object_id) -> bool

    # ── 附着/脱离 (双层: MoveIt 层 gobject 同步) ──
    def attach_object(self, object_id, link, touch_links) -> bool
    def detach_object(self, object_id, link) -> bool

    # ── 查询 ──
    def get_current_scene() -> PlanningScene
```

参数说明：
- `shape_type`: `SolidPrimitive.BOX | SPHERE | CYLINDER`
- `dimensions`: `[x, y, z]` 或 `[radius, height]`（m）
- `frame_id`: 位姿参考坐标系（调用方传 vlm_bridge 输出的 frame_id）

## 4. grab_action.py 改造

### 4.1 状态机

```
IDLE
  │ 收到 /target_pre_grasp
  ▼
SETUP
  │ psm.add_object("target", BOX, [0.03,0.03,0.03], pose, "base")
  │ _publish_status("stage1_done")  → 触发 vlm_bridge stage 2
  │
  ▼
PRE_GRASP_PLAN ──失败──→ FAILED
  │ MoveIt 规划：物体在 planning scene → 自动绕行到目标上方
  ▼
PRE_GRASP_EXEC
  │
  ▼
AWAIT_STAGE2  ← 等 /target_pose，10s 超时降级
  │
  ▼
APPROACH:
  ① _send_gripper_command(open)             ← 此时离目标仅 8cm，张爪不会引入规划失败
  ② psm.allow_collision("gripper", "target")  ← 或 psm.remove_object("target") fallback
  ③ MoveIt 规划 Cartesian 下移                 ← MoveIt 不再避让
  │
  ▼
APPROACH_EXEC
  │
  ▼
GRASP
  │ _send_gripper_command(close)
  │ Gazebo: AttachLink  (物理粘住)
  │ MoveIt: psm.attach_object("target", "gripper", touch_links)  (规划附着)
  │ _publish_status("stage2_done")
  ▼
LIFT_PLAN
  │ 物体已附着 → MoveIt 当 robot 一部分 → 自动防碰
  ▼
LIFT_EXEC
  │
  ▼
TRANSPORT_PLAN  ──→ 自由规划到放置位姿
  ▼
TRANSPORT_EXEC
  │
  ▼
PLACE
  │ Gazebo: DetachLink
  │ MoveIt: psm.detach_object("target", "gripper")
  │ _send_gripper_command(open)
  │ psm.forbid_collision("gripper", "target")
  │ psm.remove_object("target")
  │ _publish_status("grasp_complete")
  ▼
IDLE
```

### 4.2 关键设计决策

**张爪时机**：SETUP 阶段夹爪保持闭合（缩小扫掠体积，提高 pre-grasp 规划成功率），APPROACH 阶段再张爪。此时离目标仅 pre_grasp_z_offset（0.08m），直线下移的张爪不会引入规划失败。与 MTC 标准做法不同但适配 SO101 的小工作空间。

**双层同步**：每个涉及物体状态变更的阶段都同时操作 Gazebo（LinkAttacher）和 MoveIt（PlanningSceneManager），两层保持一致。

**Approach 碰撞策略**：先尝试 `allow_collision`（MTC 式，保留物体信息），失败则 fallback 到 `remove_object`。提供参数 `approach_mode` 切换：`"allow" | "remove" | "auto"`（默认 auto，自动 fallback）。

### 4.3 参数新增

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `approach_mode` | `"auto"` | 碰撞处理策略: `allow` / `remove` / `auto` |
| `target_object_shape` | `"BOX"` | 目标物体形状 |
| `target_object_dims` | `[0.03, 0.03, 0.03]` | 目标物体尺寸 (m) |

### 4.4 错误恢复

- Planning Scene 操作失败（service 不可用）→ 退化到当前行为（静默继续，日志 warning）
- `allow_collision` 失败 → 自动 fallback 到 `remove_object`
- `remove_object` 也失败 → `FAILED` 状态

## 5. 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `position_topic/planning_scene_manager.py` | **新建** | Planning Scene 操作封装 |
| `position_topic/grab_action.py` | **重写** | 集成 PSM + 新状态机 |
| `position_topic/setup.py` | 可能修改 | 如有新依赖 |

**不变**：`vlm_bridge.py`、`camera_utils.py`、`vlm_client.py`、`prompts.py`、`position_subscriber.py`、`position_publisher.py`

## 6. 测试策略

### 6.1 单元级
- `PlanningSceneManager` 各方法：mock ApplyPlanningScene 验证请求构造正确

### 6.2 集成测试（Gazebo）
- 无遮挡方块抓取全流程：VLM → PSM add → pre-grasp → approach → grasp → lift → transport → place
- 验证物体不再被撞飞（`allow_collision` 后 MoveIt 不避让 = 路径精确到达目标）
- 验证搬运过程无碰撞（物体 attach 后受 MoveIt 保护）
- 验证 PSM 清理（place 后 scene 为空）
