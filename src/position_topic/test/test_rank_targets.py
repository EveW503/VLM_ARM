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
    assert labels == ["ripe_clear", "ripe_blocked", "overripe_clear", "unripe_clear"]


def test_rank_unknown_ripeness_treated_as_unripe():
    """未知成熟度视为 unripe (最差)。"""
    targets = [
        {"label": "a", "ripeness": "unknown_value", "obstacle_above": "none", "confidence": 0.99},
        {"label": "b", "ripeness": "ripe", "obstacle_above": "none", "confidence": 0.50},
    ]
    ranked = rank_targets(targets)
    assert ranked[0]["label"] == "b"
