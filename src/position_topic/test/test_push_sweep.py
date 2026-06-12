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
