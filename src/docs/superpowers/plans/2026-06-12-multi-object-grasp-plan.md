# 多目标软硬识别与顺序采摘 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 VLM 单目标抓取升级为"识别所有目标 → 软硬分类排序 → 推扫排障 → 循环逐个采摘直到无果实"的完整采摘流程。

**Architecture:** 6 个任务，按依赖顺序：增强 VLM Prompt → 纯函数排序 → 推扫轨迹生成 → vlm_bridge 多目标循环逻辑 → grab_action PUSH_SWEEP 状态 → 集成配置。每个任务可独立测试。

**Tech Stack:** Python 3, ROS 2 Humble, Qwen3-VL-Plus (DashScope), MoveIt 2, Gazebo Classic

---

### Task 1: Rewrite STAGE1_SYSTEM_PROMPT for multi-target output

**Files:**
- Modify: `position_topic/position_topic/prompts.py:7-33`

- [ ] **Step 1: Write the new STAGE1_SYSTEM_PROMPT**

Replace `STAGE1_SYSTEM_PROMPT` (lines 7-33) with the multi-target version:

```python
STAGE1_SYSTEM_PROMPT = """\
你是一个草莓采摘机器人的视觉系统。你的任务是分析俯视视角的农业场景图像。

请完成以下任务:
1. 识别图像中所有可见的草莓果实，逐个标注属性
2. 识别图像中所有障碍物（茎杆、叶片等非果实物体）

对每个草莓果实，标注以下属性:
- material: 果实始终是 "soft"
- ripeness: 成熟度——"ripe"(红色全熟)、"unripe"(青色未熟)、"overripe"(深红过熟)
- obstacle_above: 该果实正上方是否有遮挡——"none"(无遮挡)、"leaf"(叶片遮挡)、"stem"(茎杆遮挡)
- optimal_approach_direction: 从哪个方向接近果实视线最清晰——'front'|'left'|'right'|'front_left'|'front_right'|'top'

请严格按以下 JSON 格式回复，不要包含任何其他文字:
{
  "targets": [
    {
      "bbox": [<整数, 左上角x>, <整数, 左上角y>, <整数, 右下角x>, <整数, 右下角y>],
      "label": "<描述, 如 'strawberry'>",
      "confidence": <0.0-1.0之间的浮点数>,
      "material": "soft",
      "ripeness": "<'ripe'|'unripe'|'overripe'>",
      "obstacle_above": "<'none'|'leaf'|'stem'>",
      "optimal_approach_direction": "<'front'|'left'|'right'|'front_left'|'front_right'|'top'>"
    }
  ],
  "obstacles": [
    {
      "bbox": [<整数, 左上角x>, <整数, 左上角y>, <整数, 右下角x>, <整数, 右下角y>],
      "label": "<描述, 如 'leaf' 或 'stem'>",
      "material": "<'soft'|'hard'>",
      "confidence": <0.0-1.0之间的浮点数>
    }
  ]
}

注意:
- bbox 坐标使用归一化坐标 [0, 1000], (0,0)=左上角, (1000,1000)=右下角
- bbox 应紧密包围物体，不要留太大余量
- 忽略画面中可能出现的机械臂部件
- targets 列表按抓取优先级从高到低排列
- 如果没有发现任何可采摘的草莓果实, targets 设为空数组 []
- 如果没有发现任何障碍物, obstacles 设为空数组 []
- 最多返回 10 个果实目标
"""
```

- [ ] **Step 2: Verify the new prompt string is syntactically valid**

Run: `python3 -c "from position_topic.prompts import STAGE1_SYSTEM_PROMPT; print('OK:', len(STAGE1_SYSTEM_PROMPT), 'chars'); assert 'targets' in STAGE1_SYSTEM_PROMPT; assert 'obstacles' in STAGE1_SYSTEM_PROMPT; assert 'material' in STAGE1_SYSTEM_PROMPT; assert 'ripeness' in STAGE1_SYSTEM_PROMPT; assert 'obstacle_above' in STAGE1_SYSTEM_PROMPT; print('All assertions passed')"`

Expected: All assertions passed

- [ ] **Step 3: Commit**

```bash
cd ~/lerobot_ws/src
git add position_topic/position_topic/prompts.py
git commit -m "feat: rewrite STAGE1 prompt for multi-target + soft/hard + ripeness output"
```

---

### Task 2: Add rank_targets() sort function to vlm_bridge.py

**Files:**
- Modify: `position_topic/position_topic/vlm_bridge.py` (add function at top, before VlmBridge class)

- [ ] **Step 1: Add rank_targets() function**

Add after the imports (after line 29) and before the `VlmBridge` class (before line 31):

```python
# ── 目标排序 ─────────────────────────────────────────

RIPENESS_ORDER = {"ripe": 0, "overripe": 1, "unripe": 2}
OBSTACLE_ORDER = {"none": 0, "leaf": 1, "stem": 2}


def rank_targets(targets):
    """
    对 VLM 返回的目标列表按优先级排序。

    排序 key: 成熟度(ripe优先) > 上方无障碍 > 置信度高 > 原始索引

    Args:
        targets: list[dict], 每个 dict 包含:
            ripeness (str), obstacle_above (str), confidence (float)

    Returns:
        list[dict]: 排序后的目标列表 (最优在前)
    """
    def _sort_key(item):
        t, idx = item
        ripeness_score = RIPENESS_ORDER.get(t.get("ripeness", "unripe"), 2)
        obstacle_score = OBSTACLE_ORDER.get(t.get("obstacle_above", "leaf"), 2)
        # 置信度取负值 → 高置信度排前面
        return (ripeness_score, obstacle_score, -t.get("confidence", 0.0), idx)

    indexed = list(enumerate(targets))
    indexed.sort(key=_sort_key)
    return [t for _, t in indexed]
```

- [ ] **Step 2: Write unit test**

Write to `position_topic/test/test_rank_targets.py`:

```python
"""Tests for rank_targets function."""
import pytest
from position_topic.vlm_bridge import rank_targets


def test_rank_ripe_before_unripe():
    """完全成熟优先于未成熟，即使未成熟置信度更高。"""
    targets = [
        {"label": "a", "ripeness": "unripe", "obstacle_above": "none", "confidence": 0.99},
        {"label": "b", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.50},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
    assert ranked[1]["label"] == "a"


def test_rank_no_obstacle_before_leaf():
    """无障碍优先于叶片遮挡。"""
    targets = [
        {"label": "a", "ripeness": "ripe", "obstacle_above": "leaf", "confidence": 0.99},
        {"label": "b", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.50},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
    assert ranked[1]["label"] == "a"


def test_rank_higher_confidence_tiebreaker():
    """同等条件下，高置信度优先。"""
    targets = [
        {"label": "a", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.60},
        {"label": "b", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.90},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
    assert ranked[1]["label"] == "a"


def test_rank_stem_worse_than_leaf():
    """茎杆遮挡比叶片遮挡更差——stem 排在 leaf 后面。"""
    targets = [
        {"label": "a", "ripeness": "ripe", "obstacle_above": "stem", "confidence": 0.99},
        {"label": "b", "ripeness": "ripe", "obstacle_above": "leaf", "confidence": 0.50},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
    assert ranked[1]["label"] == "a"


def test_rank_empty_list():
    """空列表返回空列表。"""
    assert rank_targets([]) == []


def test_rank_single_target():
    """单目标直接返回。"""
    targets = [{"label": "only", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.50}]
    ranked = rank_targets(targets)
    assert len(ranked) == 1
    assert ranked[0]["label"] == "only"


def test_rank_overripe_before_unripe():
    """过熟优先于未熟（避免腐烂）。"""
    targets = [
        {"label": "a", "ripeness": "unripe", "obstacle_above": "none", "confidence": 0.90},
        {"label": "b", "ripeness": "overripe", "obstacle_above": "none", "confidence": 0.50},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
    assert ranked[1]["label"] == "a"


def test_rank_full_priority_chain():
    """完整优先级链: ripe+none > ripe+leaf > overripe+none > unripe+none。"""
    targets = [
        {"label": "unripe_clear", "ripeness": "unripe", "obstacle_above": "none", "confidence": 0.99},
        {"label": "ripe_blocked", "ripeness": "ripe", "obstacle_above": "leaf", "confidence": 0.99},
        {"label": "ripe_clear", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.80},
        {"label": "overripe_clear", "ripeness": "overripe", "obstacle_above": "none", "confidence": 0.90},
    ]
    ranked = rank_targets(targets)
    labels = [t["label"] for t in ranked]
    assert labels == ["ripe_clear", "overripe_clear", "ripe_blocked", "unripe_clear"]


def test_rank_unknown_ripeness_treated_as_unripe():
    """未知成熟度视为 unripe (最差)。"""
    targets = [
        {"label": "a", "ripeness": "unknown_value", "obstacle_above": "none", "confidence": 0.99},
        {"label": "b", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.50},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/lerobot_ws && python3 -m pytest src/position_topic/test/test_rank_targets.py -v`

Expected: 2 tests pass (empty + single), 7 tests fail with import error — `rank_targets` is not yet importable from the correct path. Actually, since rank_targets is defined IN vlm_bridge.py and vlm_bridge imports ROS modules, this won't import cleanly outside ROS. Let's test the function in isolation instead.

Run: `cd ~/lerobot_ws/src && python3 -c "
# Extract and test rank_targets in isolation
exec(open('position_topic/position_topic/vlm_bridge.py').read().split('class VlmBridge')[0].split('# ── 目标排序 ──')[1].split('def rank_targets')[0] + open('position_topic/position_topic/vlm_bridge.py').read().split('class VlmBridge')[0])
print('rank_targets imported successfully')
"`

This is fragile. Better approach: write a small test script that imports just the function after setting up a mock for the ROS imports.

Actually, the cleanest approach: since `rank_targets` is a pure function with no ROS dependencies, write it as a separate import that both the test and vlm_bridge can use. But the spec says it goes in vlm_bridge.py.

Let me use a simpler test approach — test via subprocess that mocks the ROS imports:

Run: `cd ~/lerobot_ws/src && python3 -c "
import sys, unittest

# Mock ROS modules before importing
import types
rclpy_mod = types.ModuleType('rclpy')
rclpy_mod.node = types.ModuleType('rclpy.node')
rclpy_mod.node.Node = object
sys.modules['rclpy'] = rclpy_mod
sys.modules['rclpy.node'] = rclpy_mod.node

# Now we can't import the full module due to other ROS deps.
# Instead, paste the function source and test it directly.
exec('''$(sed -n '/^RIPENESS_ORDER/,/^class VlmBridge/p' position_topic/position_topic/vlm_bridge.py | head -n -1)''')

# Run tests
targets = [
    {'label': 'a', 'ripeness': 'unripe', 'obstacle_above': 'none', 'confidence': 0.99},
    {'label': 'b', 'ripeness': 'ripe', 'obstacle_above': 'none', 'confidence': 0.50},
]
ranked = rank_targets(targets)
assert ranked[0]['label'] == 'b', f'Expected b first, got {ranked[0][\"label\"]}'
assert ranked[1]['label'] == 'a', f'Expected a second, got {ranked[1][\"label\"]}'
print('All inline tests passed')
"`

Expected: `All inline tests passed`

- [ ] **Step 4: Commit**

```bash
cd ~/lerobot_ws/src
git add position_topic/position_topic/vlm_bridge.py position_topic/test/test_rank_targets.py
git commit -m "feat: add rank_targets() sort function with priority chain"
```

---

### Task 3: Create push_sweep.py trajectory generator

**Files:**
- Create: `position_topic/position_topic/push_sweep.py`

- [ ] **Step 1: Write push_sweep.py**

```python
"""
推扫排障轨迹生成模块。

生成水平推扫 Cartesian 航点列表，用于推开目标上方的软障碍物（叶片）。
硬障碍物不适用推扫——由调用方判断 material 后决定是否调用本模块。
"""
import math

from geometry_msgs.msg import Pose, Point, Quaternion


# 推扫方向 → (Y 偏移量, Z 偏移量) in base frame
#   - Y: 水平推扫方向 (正=右, 负=左)
#   - Z: 垂直偏移 (正=上)
SWEEP_DIRECTION_OFFSETS = {
    "front":       ( 0.00,  0.05),  # 正前方接近，不推
    "left":        ( 0.08,  0.05),  # 从左接近 → 向右推 8cm
    "right":       (-0.08,  0.05),  # 从右接近 → 向左推 8cm
    "front_left":  ( 0.06,  0.05),
    "front_right": (-0.06,  0.05),
    "top":         ( 0.00,  0.05),  # 顶部接近不推
}

# 推扫末端朝向: Ry(-90°) — gripper X 指向世界 -Z (垂直向下)
# 与 PRE_GRASP 使用相同朝向
_SWEEP_SQRT2_2 = math.sqrt(2) / 2.0
SWEEP_ORIENTATION = Quaternion(
    x=0.0, y=-_SWEEP_SQRT2_2, z=0.0, w=_SWEEP_SQRT2_2,
)


def generate_sweep_waypoints(pre_grasp_pose, approach_direction):
    """
    生成推扫 Cartesian 航点列表。

    轨迹: 起点(目标正上方) → 水平推扫点 → 返回起点
    三步走模拟"用手拨开叶子"。

    Args:
        pre_grasp_pose: geometry_msgs/Pose, PRE_GRASP 位姿 (目标正上方)
        approach_direction: str, 枚举值: 'front'|'left'|'right'|'front_left'|'front_right'|'top'

    Returns:
        list[geometry_msgs/Pose]: 推扫航点列表 (3 个点),
        如果方向不触发推扫 (front/top), 返回空列表 []
    """
    direction = approach_direction.lower()
    if direction not in SWEEP_DIRECTION_OFFSETS:
        return []

    y_offset, z_offset = SWEEP_DIRECTION_OFFSETS[direction]

    # front 和 top 方向不推扫——直接下探即可
    if abs(y_offset) < 1e-6:
        return []

    px = pre_grasp_pose.position.x
    py = pre_grasp_pose.position.y
    pz = pre_grasp_pose.position.z

    # 航点 1: 起点 (目标正上方 + z_offset)
    wp_start = Pose()
    wp_start.position = Point(x=px, y=py, z=pz + z_offset)
    wp_start.orientation = SWEEP_ORIENTATION

    # 航点 2: 水平推扫点
    wp_sweep = Pose()
    wp_sweep.position = Point(x=px, y=py + y_offset, z=pz + z_offset)
    wp_sweep.orientation = SWEEP_ORIENTATION

    # 航点 3: 返回起点
    wp_return = Pose()
    wp_return.position = Point(x=px, y=py, z=pz + z_offset)
    wp_return.orientation = SWEEP_ORIENTATION

    return [wp_start, wp_sweep, wp_return]


def should_push_sweep(obstacle_above, material):
    """
    判断是否需要推扫排障。

    仅当同时满足以下条件时返回 True:
    1. obstacle_above != "none" — 有障碍物在上方
    2. material == "soft" — 障碍物是软的，可以被推开

    Args:
        obstacle_above: str, 枚举值: 'none'|'leaf'|'stem'
        material: str, 枚举值: 'soft'|'hard'

    Returns:
        bool: 是否应执行推扫
    """
    return obstacle_above != "none" and material == "soft"
```

- [ ] **Step 2: Write unit test**

Write to `position_topic/test/test_push_sweep.py`:

```python
"""Tests for push_sweep module."""
import math
import pytest
from geometry_msgs.msg import Pose, Point, Quaternion
from position_topic.push_sweep import (
    generate_sweep_waypoints,
    should_push_sweep,
    SWEEP_DIRECTION_OFFSETS,
)


def _make_pose(x, y, z):
    p = Pose()
    p.position = Point(x=x, y=y, z=z)
    return p


def test_generate_sweep_waypoints_left_direction():
    """left 方向生成 3 个航点，Y 偏移为正。"""
    pre_grasp = _make_pose(0.2, 0.0, 0.15)
    waypoints = generate_sweep_waypoints(pre_grasp, "left")
    assert len(waypoints) == 3

    # 航点 1: 起点
    assert abs(waypoints[0].position.x - 0.2) < 1e-6
    assert abs(waypoints[0].position.y - 0.0) < 1e-6
    assert waypoints[0].position.z > 0.15  # 加了 z_offset

    # 航点 2: 推扫点 → Y 偏移 +0.08
    assert abs(waypoints[1].position.y - 0.08) < 1e-6
    # X 不变
    assert abs(waypoints[1].position.x - 0.2) < 1e-6

    # 航点 3: 返回起点
    assert abs(waypoints[2].position.y - 0.0) < 1e-6
    assert abs(waypoints[2].position.x - 0.2) < 1e-6


def test_generate_sweep_waypoints_right_direction():
    """right 方向 Y 偏移为负。"""
    pre_grasp = _make_pose(0.25, 0.05, 0.20)
    waypoints = generate_sweep_waypoints(pre_grasp, "right")
    assert len(waypoints) == 3
    assert waypoints[1].position.y < 0.05  # 负偏移


def test_generate_sweep_waypoints_front_returns_empty():
    """front 方向不触发推扫，返回空列表。"""
    pre_grasp = _make_pose(0.2, 0.0, 0.15)
    waypoints = generate_sweep_waypoints(pre_grasp, "front")
    assert waypoints == []


def test_generate_sweep_waypoints_top_returns_empty():
    """top 方向不触发推扫，返回空列表。"""
    pre_grasp = _make_pose(0.2, 0.0, 0.15)
    waypoints = generate_sweep_waypoints(pre_grasp, "top")
    assert waypoints == []


def test_generate_sweep_waypoints_unknown_direction_returns_empty():
    """未知方向返回空列表。"""
    pre_grasp = _make_pose(0.2, 0.0, 0.15)
    waypoints = generate_sweep_waypoints(pre_grasp, "unknown_dir")
    assert waypoints == []


def test_generate_sweep_waypoints_all_have_orientation():
    """所有航点都有正确的朝向。"""
    pre_grasp = _make_pose(0.2, 0.0, 0.15)
    for direction in ["left", "right", "front_left", "front_right"]:
        waypoints = generate_sweep_waypoints(pre_grasp, direction)
        for wp in waypoints:
            assert wp.orientation.y != 0.0 or wp.orientation.w != 0.0


def test_generate_sweep_waypoints_z_offset_applied():
    """航点 Z 应高于 pre_grasp (加了偏移)。"""
    pre_grasp = _make_pose(0.2, 0.0, 0.10)
    waypoints = generate_sweep_waypoints(pre_grasp, "left")
    for wp in waypoints:
        assert wp.position.z > 0.10


def test_should_push_sweep_leaf_soft():
    """叶片遮挡 + soft → 应该推扫。"""
    assert should_push_sweep("leaf", "soft") is True


def test_should_push_sweep_none():
    """无遮挡 → 不推扫。"""
    assert should_push_sweep("none", "soft") is False


def test_should_push_sweep_stem_hard():
    """茎杆遮挡 + hard → 不推扫。"""
    assert should_push_sweep("stem", "hard") is False


def test_should_push_sweep_stem_soft():
    """茎杆但 material=soft → 推扫（按 material 判断，不按 label）。"""
    assert should_push_sweep("stem", "soft") is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/lerobot_ws && colcon test --packages-select position_topic --event-handlers console_direct+`

Expected: Some tests fail because `push_sweep` module doesn't exist yet. Existing boilerplate tests (flake8/copyright) pass.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/lerobot_ws && python3 -m pytest src/position_topic/test/test_push_sweep.py -v`

Expected: All 11 tests PASS. (Tests use geometry_msgs which is available in the ROS environment.)

But `geometry_msgs` might not be on PYTHONPATH outside ROS. Use:

Run: `cd ~/lerobot_ws && source install/setup.bash && python3 -m pytest src/position_topic/test/test_push_sweep.py -v`

Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
cd ~/lerobot_ws/src
git add position_topic/position_topic/push_sweep.py position_topic/test/test_push_sweep.py
git commit -m "feat: add push_sweep module for soft obstacle clearing"
```

---

### Task 4: Update vlm_bridge.py — multi-target parsing + metadata + loop control

**Files:**
- Modify: `position_topic/position_topic/vlm_bridge.py` (multiple sections)
- Modify: `position_topic/position_topic/vlm_client.py:47` (max_tokens default)

**Context:** vlm_bridge.py currently:
- `_run_stage1()` (line 234): parses single `target` from VLM, publishes one target
- `_run_stage2()` (line 369): publishes precise target
- `_grab_status_cb()` (line 155): handles "stage1_done" and "error"
- State machine: IDLE → STAGE1_QUERY → STAGE1_WAIT → STAGE2_QUERY → IDLE

Changes needed:
1. `_run_stage1()`: parse `targets[]` array, store in `self._targets_pool`, sort with `rank_targets`, pick best, publish `/target_metadata`
2. New `/target_metadata` publisher
3. `_grab_status_cb()`: handle "grasp_complete" → re-trigger Stage1; handle "error" → try next target from pool or re-trigger Stage1
4. Blacklist + fail count tracking for failed targets
5. Termination detection

- [ ] **Step 1: Add /target_metadata publisher to __init__**

In `VlmBridge.__init__()`, after the `/target_observation_pose` publisher setup (after line 103), add:

```python
        # --- 目标元数据发布 (材质、遮挡信息等) ---
        self._metadata_pub = self.create_publisher(
            String, "/target_metadata", 10
        )
        # --- 任务状态发布 (task_complete) ---
        self._task_status_pub = self.create_publisher(
            String, "/task_status", 10
        )
```

- [ ] **Step 2: Add multi-target state variables to __init__**

In `VlmBridge.__init__()`, after the `_state_timer` line (after line 111), add:

```python
        # --- 多目标状态 ---
        self._targets_pool = []       # 当前轮次排序后的目标列表
        self._current_target_idx = 0  # 当前选中目标在 pool 中的索引
        self._target_fail_counts = {}  # {label: fail_count} 跟踪每个目标失败次数
        self._target_blacklist = set()  # 黑名单目标标签集合
        self._consecutive_empty_stage1 = 0  # 连续空 Stage1 计数
```

- [ ] **Step 3: Rewrite _run_stage1() to parse targets[] array**

Replace the Stage1 result parsing section (lines 252-265, the part that reads `target = result.get("target", {})`) with:

```python
            # 解析多目标列表
            targets = result.get("targets", [])
            obstacles = result.get("obstacles", [])

            if not targets:
                self.get_logger().warning(
                    f"VLM 未发现可采摘目标 (targets=[]), obstacles={len(obstacles)} 个"
                )
                self._consecutive_empty_stage1 += 1
                if self._consecutive_empty_stage1 >= 2:
                    self.get_logger().info("连续 2 次无目标, 任务完成")
                    self._publish_task_complete()
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
                return
            self._consecutive_empty_stage1 = 0

            self.get_logger().info(
                f"Stage1 发现 {len(targets)} 个目标, {len(obstacles)} 个障碍物"
            )

            # 排序: 成熟度 > 无障碍 > 置信度
            sorted_targets = rank_targets(targets)
            self._targets_pool = sorted_targets
            self._current_target_idx = 0
```

- [ ] **Step 4: Add target selection + metadata publishing logic to _run_stage1()**

After the sorting code from Step 3, replace the single-target processing (lines 267-359, from `if bbox[0] < 0:` through the end of `_run_stage1` except the exception handler) with a call to a new method `_process_current_target()`:

```python
            # 处理当前最优目标
            success = self._process_current_target(rgb, depth, info)
            if not success:
                with self._lock:
                    self._state = self.STAGE_IDLE
                    self._vlm_in_progress = False
```

Then add the new method `_process_current_target()` before `_run_stage2()`:

```python
    def _process_current_target(self, rgb, depth, info):
        """
        从当前 targets_pool 中选取当前索引的目标，执行 2D→3D 映射并发布。

        Returns:
            bool: True 如果目标处理成功并已发布
        """
        if self._current_target_idx >= len(self._targets_pool):
            self.get_logger().warning("targets_pool 已耗尽, 需要重新 Stage1")
            return False

        target = self._targets_pool[self._current_target_idx]
        bbox = target.get("bbox", [-1, -1, -1, -1])
        confidence = target.get("confidence", 0.0)
        label = target.get("label", "unknown")
        direction = target.get("optimal_approach_direction", "top")
        material = target.get("material", "soft")
        ripeness = target.get("ripeness", "unripe")
        obstacle_above = target.get("obstacle_above", "none")

        if bbox[0] < 0:
            self.get_logger().warning(f"目标 {label} bbox 无效, 跳过")
            self._current_target_idx += 1
            return self._process_current_target(rgb, depth, info)

        # Qwen3-VL bbox 坐标转换
        x1 = int(bbox[0] / 1000.0 * rgb.width)
        y1 = int(bbox[1] / 1000.0 * rgb.height)
        x2 = int(bbox[2] / 1000.0 * rgb.width)
        y2 = int(bbox[3] / 1000.0 * rgb.height)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        self.get_logger().info(
            f"选中目标 [{self._current_target_idx}/{len(self._targets_pool)}]: "
            f"{label} ripeness={ripeness} material={material} "
            f"obstacle={obstacle_above} dir={direction} conf={confidence:.2f}"
        )

        Z = get_min_depth_in_region(depth, x1, y1, x2, y2)
        if Z is None:
            self.get_logger().error(f"目标 {label} 深度查询失败")
            self._current_target_idx += 1
            return self._process_current_target(rgb, depth, info)

        X_cam, Y_cam, Z_cam = pixel_to_camera_3d(cx, cy, Z, info)
        pt = transform_point(
            self._tf_buffer,
            X_cam, Y_cam, Z_cam,
            "camera_depth_optical_frame", "base",
        )
        if pt is None:
            self.get_logger().error(f"目标 {label} TF 变换失败")
            self._current_target_idx += 1
            return self._process_current_target(rgb, depth, info)

        p_rough = pt
        self.get_logger().info(
            f"3D rough (base): ({p_rough[0]:.4f}, {p_rough[1]:.4f}, {p_rough[2]:.4f})"
        )

        # 方位偏移 → 观察点
        DIRECTION_OFFSETS = {
            "front":       (-0.06,  0.00,  0.10),
            "left":        ( 0.00,  0.08,  0.10),
            "right":       ( 0.00, -0.08,  0.10),
            "front_left":  (-0.05,  0.06,  0.10),
            "front_right": (-0.05, -0.06,  0.10),
            "top":         ( 0.00,  0.00,  0.15),
        }
        offset = DIRECTION_OFFSETS.get(direction, DIRECTION_OFFSETS["top"])

        obs_x = p_rough[0] + offset[0]
        obs_y = p_rough[1] + offset[1]
        obs_z = p_rough[2] + offset[2]

        if obs_x < 0.10:
            self.get_logger().warn(
                f"观察点 X={obs_x:.3f} 进入基座死区, 强制钳制至 X=0.10"
            )
            obs_x = 0.10

        p_obs = (obs_x, obs_y, obs_z)
        q = compute_lookat_quaternion(p_obs, p_rough)
        if q is None:
            self.get_logger().error("LookAt 四元数计算失败, 降级为 identity")
            q = (0.0, 0.0, 0.0, 1.0)

        self.get_logger().info(
            f"观察位姿 (base): pos=({p_obs[0]:.4f}, {p_obs[1]:.4f}, {p_obs[2]:.4f})"
        )

        with self._lock:
            if not self._dry_run:
                # 发布观察位姿
                self._pending_publish = (
                    "observation",
                    Point(x=p_obs[0], y=p_obs[1], z=p_obs[2]),
                    q,
                )
                # 发布粗定位
                self._pending_rough = (
                    "pre_grasp",
                    Point(x=p_rough[0], y=p_rough[1], z=p_rough[2]),
                )
            self._state = self.STAGE1_WAIT
            self._vlm_in_progress = False

        # 发布元数据 (独立于 dry_run，方便调试)
        if not self._dry_run:
            self._publish_metadata(material, obstacle_above, direction, ripeness, label)

        return True

    def _publish_metadata(self, material, obstacle_above, direction, ripeness, label):
        """发布当前目标的语义元数据到 /target_metadata。"""
        import json
        meta = {
            "material": material,
            "obstacle_above": obstacle_above,
            "optimal_approach_direction": direction,
            "ripeness": ripeness,
            "label": label,
        }
        msg = String()
        msg.data = json.dumps(meta, ensure_ascii=False)
        self._metadata_pub.publish(msg)
        self.get_logger().info(f"已发布 /target_metadata: {msg.data}")

    def _publish_task_complete(self):
        """发布任务完成信号。"""
        msg = String()
        msg.data = "task_complete"
        self._task_status_pub.publish(msg)
        self.get_logger().info("已发布 /task_status: task_complete — 采摘任务结束")
```

- [ ] **Step 5: Update _grab_status_cb() for loop control**

Replace the `_grab_status_cb` method (lines 155-165) with:

```python
    def _grab_status_cb(self, msg):
        import json
        with self._lock:
            if msg.data == "stage1_done" and self._state == self.STAGE1_WAIT:
                self.get_logger().info("收到 stage1_done, 触发阶段二推理")
                self._state = self.STAGE2_QUERY

            elif msg.data == "grasp_complete":
                self.get_logger().info("收到 grasp_complete, 重新触发阶段一")
                self._targets_pool = []
                self._current_target_idx = 0
                self._target_fail_counts.clear()
                self._state = self.STAGE1_QUERY

            elif msg.data == "error":
                self.get_logger().warning("收到抓取失败(error)，尝试下一个目标")
                self._handle_grab_error()

    def _handle_grab_error(self):
        """处理抓取失败: 递增失败计数，如果超阈值则跳过该目标。"""
        if self._current_target_idx < len(self._targets_pool):
            target = self._targets_pool[self._current_target_idx]
            label = target.get("label", "unknown")
            # 用 bbox 中心点作为唯一标识
            bbox = target.get("bbox", [0, 0, 0, 0])
            target_key = f"{label}_{bbox[0]}_{bbox[1]}"

            count = self._target_fail_counts.get(target_key, 0) + 1
            self._target_fail_counts[target_key] = count
            self.get_logger().warning(
                f"目标 '{label}' 失败 {count}/3 次"
            )
            if count >= 3:
                self.get_logger().error(
                    f"目标 '{label}' 连续失败 3 次, 加入黑名单"
                )
                self._target_blacklist.add(target_key)
                self._current_target_idx += 1

                if self._current_target_idx >= len(self._targets_pool):
                    self.get_logger().warning("所有目标已尝试, 重新 Stage1")
                    self._targets_pool = []
                    self._current_target_idx = 0
                    self._state = self.STAGE1_QUERY
                else:
                    # 重试下一个目标——需要重新获取相机帧
                    self.get_logger().info("尝试下一个目标")
                    self._state = self.STAGE1_QUERY
            else:
                # 重试同一目标——重新获取相机帧
                self.get_logger().info(f"重试同一目标 '{label}' ({count}/3)")
                self._state = self.STAGE1_QUERY
        else:
            self._state = self.STAGE_IDLE
        self._vlm_in_progress = False
        self._pending_publish = None
        self._pending_rough = None
```

- [ ] **Step 6: Increase VLM max_tokens for Stage1**

In `vlm_client.py` line 47, change the default:

```python
    def call_vlm(self, system_prompt, user_instruction, image_data_url, max_tokens=500):
```

And in `_run_stage1()`, explicitly pass the larger token limit. Find the call on line 242-244:

```python
            result = self._vlm.call_vlm(
                STAGE1_SYSTEM_PROMPT, self._task_instruction, img_b64,
                max_tokens=500,
            )
```

- [ ] **Step 7: Import rank_targets in vlm_bridge.py**

Ensure `rank_targets` is available. Since it's defined in the same file above the class (Task 2), no import needed. Verify it's placed before the `VlmBridge` class definition.

- [ ] **Step 8: Verify the updated module syntax**

Run: `cd ~/lerobot_ws/src && python3 -c "import ast; ast.parse(open('position_topic/position_topic/vlm_bridge.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 9: Commit**

```bash
cd ~/lerobot_ws/src
git add position_topic/position_topic/vlm_bridge.py position_topic/position_topic/vlm_client.py
git commit -m "feat: vlm_bridge multi-target parsing, /target_metadata, loop control"
```

---

### Task 5: Update grab_action.py — PUSH_SWEEP state + metadata subscription + failure tracking

**Files:**
- Modify: `position_topic/position_topic/grab_action.py` (multiple sections)

**Context:** grab_action.py currently:
- 10 states: IDLE → SETUP → MOVE_TO_OBSERVE → AWAIT_STAGE2 → PRE_GRASP → APPROACH → GRASP → LIFT_PLAN → TRANSPORT_PLAN → PLACE → IDLE
- No knowledge of obstacle/material metadata
- PLACE goes to IDLE and waits (already correct for loop)

Changes needed:
1. Subscribe to `/target_metadata` and store current target attributes
2. Add `PUSH_SWEEP` state constant
3. Add `_do_push_sweep()` method
4. Modify PRE_GRASP success callback to conditionally route to PUSH_SWEEP
5. Add failure counter for per-target skip logic
6. Import `should_push_sweep` and `generate_sweep_waypoints` from push_sweep

- [ ] **Step 1: Add PUSH_SWEEP state constant and metadata storage**

In `GrabAction.__init__()`, add the state constant after line 49:

```python
    PUSH_SWEEP = "push_sweep"
```

After the `_stage2_timer` line (after line 111), add:

```python
        # --- 目标元数据 (从 /target_metadata 更新) ---
        self._target_metadata = {}  # dict with material, obstacle_above, etc.
        self._consecutive_failures = 0  # 当前目标连续失败次数
```

- [ ] **Step 2: Add /target_metadata subscription**

In `__init__()`, after the existing subscriptions (after line 80), add:

```python
        self.create_subscription(
            String, "/target_metadata", self._metadata_cb, 10
        )
```

Add the callback method before `_transition()`:

```python
    def _metadata_cb(self, msg):
        """接收当前目标的语义元数据。"""
        import json
        try:
            self._target_metadata = json.loads(msg.data)
            self.get_logger().info(
                f"收到元数据: material={self._target_metadata.get('material')}, "
                f"obstacle={self._target_metadata.get('obstacle_above')}, "
                f"direction={self._target_metadata.get('optimal_approach_direction')}"
            )
        except json.JSONDecodeError:
            self.get_logger().warning(f"无法解析 /target_metadata: {msg.data}")
```

Reset metadata when starting new target — add to `_do_setup_impl()` after line 232 (after the `add_object` call):

```python
        # 重置失败计数(新目标)
        self._consecutive_failures = 0
```

- [ ] **Step 3: Add PUSH_SWEEP to state machine _transition()**

In `_transition()` (line 162), add the PUSH_SWEEP branch after the PRE_GRASP branch (after line 169):

```python
        elif new_state == self.PUSH_SWEEP:
            self._do_push_sweep()
```

- [ ] **Step 4: Modify PRE_GRASP success callback to conditionally route to PUSH_SWEEP**

In `_plan_pre_grasp()`, change the `_send_move_request` call (lines 357-363). The success callback currently jumps directly to APPROACH. Change to:

```python
        self._send_move_request(
            pre_grasp_pose, self._grasp_target.header.frame_id or "base",
            success_callback=self._on_pre_grasp_done,
            fail_callback=lambda: self._transition(self.FAILED),
            use_orientation=True,
        )
```

Add the new callback method before `_do_approach()`:

```python
    def _on_pre_grasp_done(self):
        """PRE_GRASP 到达后，根据元数据决定是否推扫。"""
        from .push_sweep import should_push_sweep

        obstacle = self._target_metadata.get("obstacle_above", "none")
        material = self._target_metadata.get("material", "soft")

        if should_push_sweep(obstacle, material):
            self.get_logger().info(
                f"检测到软障碍物: obstacle={obstacle}, material={material} → 执行推扫"
            )
            self._transition(self.PUSH_SWEEP)
        else:
            self.get_logger().info(
                f"无需推扫: obstacle={obstacle}, material={material} → 直接 APPROACH"
            )
            self._transition(self.APPROACH)
```

- [ ] **Step 5: Implement _do_push_sweep() and its execution pipeline**

Add these methods before `_do_approach()` (before line 366):

```python
    # ── 推扫排障 ──────────────────────────────────

    def _do_push_sweep(self):
        """执行推扫排障轨迹。"""
        from .push_sweep import generate_sweep_waypoints

        if self._grasp_target is None:
            self._transition(self.FAILED)
            return

        direction = self._target_metadata.get(
            "optimal_approach_direction", "top"
        )

        # 构建 pre_grasp 位姿作为推扫参考点
        z_offset = self.get_parameter("pre_grasp_z_offset").value
        pre_grasp_pose = Pose()
        pre_grasp_pose.position = Point(
            x=self._grasp_target.pose.position.x,
            y=self._grasp_target.pose.position.y,
            z=self._grasp_target.pose.position.z + z_offset,
        )

        waypoints = generate_sweep_waypoints(pre_grasp_pose, direction)
        if not waypoints:
            self.get_logger().info("推扫方向无需推扫, 跳过")
            self._transition(self.APPROACH)
            return

        self.get_logger().info(
            f"PUSH_SWEEP: {len(waypoints)} 个航点, direction={direction}"
        )
        self._execute_sweep_trajectory(waypoints)

    def _execute_sweep_trajectory(self, waypoints):
        """通过 Cartesian path 规划并执行推扫轨迹。"""
        if not self._cartesian_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("compute_cartesian_path 不可用 (推扫)")
            self._transition(self.APPROACH)  # 降级
            return

        req = GetCartesianPath.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.header.frame_id = "base"
        req.group_name = "arm"
        req.link_name = "gripper"
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self._cartesian_cli.call_async(req)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if future.done():
                break
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            self.get_logger().error("compute_cartesian_path 超时 (推扫)")
            self._transition(self.APPROACH)  # 降级
            return

        resp = future.result()
        if resp is None or resp.error_code.val != 1:
            self.get_logger().warning(
                f"推扫 Cartesian 规划失败, 降级为直接 APPROACH. "
                f"fraction={resp.fraction if resp else 0.0:.2f}"
            )
            self._transition(self.APPROACH)  # 降级
            return

        self.get_logger().info(
            f"推扫 Cartesian 规划成功, fraction={resp.fraction:.2f}"
        )
        # 复用现有 Cartesian 执行管道
        self._execute_cartesian_trajectory(resp.solution.joint_trajectory)
```

Modify `_on_cartesian_exec_done` to handle sweep completion. Currently it calls `_on_cartesian_result` which calls `_on_approach_done`. But for sweep, success should route to APPROACH. Add a flag to distinguish:

In `__init__()`, after metadata initialization, add:

```python
        self._sweep_in_progress = False
```

In `_execute_sweep_trajectory()`, set the flag before sending:

```python
        self._sweep_in_progress = True
```

In `_on_cartesian_result()` (line 463), modify the success path:

```python
    def _on_cartesian_result(self, future):
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f"Cartesian 执行失败: {exc}")
            self._transition(self.FAILED)
            return
        if wrapped.result.error_code == 0:
            self.get_logger().info("Cartesian 轨迹完成")
            if self._sweep_in_progress:
                self._sweep_in_progress = False
                self.get_logger().info("推扫完成, 进入 APPROACH")
                self._transition(self.APPROACH)
            else:
                self._on_approach_done()
        else:
            self.get_logger().warning(f"Cartesian 执行失败: {wrapped.result.error_string}")
            if self._sweep_in_progress:
                self._sweep_in_progress = False
                self.get_logger().warning("推扫执行失败, 降级为直接 APPROACH")
                self._transition(self.APPROACH)
            else:
                self._transition(self.FAILED)
```

- [ ] **Step 6: Add failure tracking with error feedback**

When the full grab pipeline fails (FAILED state), increment the counter and publish "error" with target info. In `_transition()`, the FAILED handler (lines 180-189), add:

```python
        elif new_state == self.FAILED:
            self.get_logger().error("抓取流程失败")
            self._consecutive_failures += 1
            self._publish_status("error")
            # ... rest of existing FAILED handler ...
```

The `_publish_status("error")` already triggers `_handle_grab_error()` in vlm_bridge (Task 4).

- [ ] **Step 7: Verify syntax**

Run: `cd ~/lerobot_ws/src && python3 -c "import ast; ast.parse(open('position_topic/position_topic/grab_action.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 8: Commit**

```bash
cd ~/lerobot_ws/src
git add position_topic/position_topic/grab_action.py
git commit -m "feat: add PUSH_SWEEP state + /target_metadata subscription to grab_action"
```

---

### Task 6: Integration — launch, build, verify

**Files:**
- Modify: `position_topic/launch/move_demo.launch.py` (add metadata-related parameter if needed)
- (No new params needed — all defaults are reasonable)

- [ ] **Step 1: Build the package**

Run: `cd ~/lerobot_ws && colcon build --symlink-install --packages-select position_topic`

Expected: Build succeeds with no errors.

- [ ] **Step 2: Verify all nodes start correctly (dry run)**

Run: `cd ~/lerobot_ws && source install/setup.bash && timeout 5 ros2 run position_topic vlm_bridge --ros-args -p dry_run:=true || true`

Expected: Node starts, outputs "VLM Bridge 节点已启动", then exits after timeout. No import errors.

Run: `cd ~/lerobot_ws && source install/setup.bash && timeout 3 ros2 run position_topic grab_action || true`

Expected: Node starts and fails to connect to MoveGroup (expected without Gazebo), but no Python import errors.

- [ ] **Step 3: Verify PUSH_SWEEP module imports correctly**

Run: `cd ~/lerobot_ws && source install/setup.bash && python3 -c "from position_topic.push_sweep import generate_sweep_waypoints, should_push_sweep; print('push_sweep OK')"`

Expected: `push_sweep OK`

- [ ] **Step 4: Verify rank_targets imports correctly**

Run: `cd ~/lerobot_ws && source install/setup.bash && python3 -c "from position_topic.vlm_bridge import rank_targets; print('rank_targets OK:', rank_targets([{'ripeness':'ripe','obstacle_above':'none','confidence':0.5}]))"`

Expected: `rank_targets OK: [{'ripeness': 'ripe', ...}]`

- [ ] **Step 5: Run full test suite**

Run: `cd ~/lerobot_ws && source install/setup.bash && python3 -m pytest src/position_topic/test/test_push_sweep.py src/position_topic/test/test_rank_targets.py -v`

Expected: All tests pass.

- [ ] **Step 6: Update CLAUDE.md with new features**

Add to `CLAUDE.md` under "已知约定":

```
- **多目标**: Stage1 VLM 输出 targets[] 数组，rank_targets() 按 成熟度>无障碍>置信度 排序
- **软硬分类**: VLM 标注 material ("soft"/"hard")，硬物体不推扫
- **推扫排障**: push_sweep 模块，3 个 Cartesian 航点水平拨开软障碍物
- **循环采摘**: grasp_complete → 重新 Stage1，直到 VLM 返回空列表
- **失败处理**: 同目标连续失败 3 次 → 黑名单跳过，尝试下一个；pool 耗尽重新 Stage1
```

And under topic 通信:

```
| `/target_metadata` | vlm_bridge→grab_action | 当前目标语义属性 (JSON: material, obstacle_above, direction, ripeness, label) |
| `/task_status` | vlm_bridge→外部 | 任务完成信号 ("task_complete") |
```

And update the grab_action state machine:

```
IDLE → SETUP → MOVE_TO_OBSERVE → AWAIT_STAGE2 → PRE_GRASP
  → [PUSH_SWEEP]  ← 仅当 obstacle_above != "none" 且 material == "soft"
  → APPROACH → GRASP → LIFT_PLAN → TRANSPORT_PLAN → PLACE → IDLE
```

- [ ] **Step 7: Commit**

```bash
cd ~/lerobot_ws/src
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with multi-target, soft/hard, push_sweep features"
```

---

## Testing Strategy

### Unit tests (pytest, no ROS)

| Test file | What it tests | Run |
|-----------|--------------|-----|
| `test/test_rank_targets.py` | 排序逻辑：成熟度、障碍物、置信度优先级 | `python3 -m pytest test/test_rank_targets.py -v` |
| `test/test_push_sweep.py` | 推扫航点生成、触发条件判断 | `python3 -m pytest test/test_push_sweep.py -v` |

### Integration tests (requires Gazebo + ROS 2)

| Scenario | How to test | Expected |
|----------|-------------|----------|
| 方块场景(向后兼容) | `ros2 launch position_topic test_no_strawberry.launch.py` → 发布 task_command | 单目标抓取，无推扫，流程正常 |
| 草莓场景(多目标) | `ros2 launch position_topic move_demo.launch.py` → 发布 task_command | VLM 返回多个目标，顺序采摘 |
| 推扫触发 | 草莓场景 + VLM 报告 obstacle_above="leaf" | PUSH_SWEEP 状态执行，3 个航点 Cartesian |
| 降级(推扫失败) | 推扫 Cartesian 规划失败 | 直接进入 APPROACH，日志 warning |
| 循环终止 | 所有草莓被采摘后 | VLM 返回空 targets → task_complete |
| 失败重试 | 某个目标故意放置在工作空间外 | 3 次失败后黑名单跳过 |
| dry_run | `--ros-args -p dry_run:=true` | VLM 调用正常，不发布消息给 grab_action |

### Common debug commands

```bash
# 查看 VLM 阶段一输出
ros2 topic echo /target_metadata --once

# 查看抓取状态流
ros2 topic echo /grab_status

# 手动触发阶段一
ros2 topic pub /task_command std_msgs/msg/String "data: '找出所有可采摘的草莓果实'"

# 查看当前目标列表日志
ros2 run rqt_console rqt_console  # GUI, 或:
tail -f ~/.ros/log/*/vlm_bridge*.log
```
