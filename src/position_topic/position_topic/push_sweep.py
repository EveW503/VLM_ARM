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
        list[geometry_msgs.Pose]: 推扫航点列表 (3 个点),
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
