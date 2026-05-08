"""
相机工具函数: 深度图查询、像素反投影、TF2 坐标变换。
"""
import numpy as np


def get_depth_at_pixel(depth_msg, u, v):
    """
    从深度图中查询指定像素的深度值 (米)。
    如果目标像素无效 (深度为 0), 回退到 3x3 邻域中值。

    Args:
        depth_msg: sensor_msgs/Image, encoding "16UC1" (mm) 或 "32FC1" (m)
        u: int, 像素列坐标
        v: int, 像素行坐标

    Returns:
        float: 深度值 (米), 或 None (查询失败)
    """
    u = int(u)
    v = int(v)

    enc = depth_msg.encoding
    if enc == "32FC1":
        data = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width
        )
        scale = 1.0  # 已经是米
        invalid = np.isnan
    elif enc == "16UC1":
        data = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
            depth_msg.height, depth_msg.width
        )
        scale = 1.0 / 1000.0  # 毫米 → 米
        invalid = lambda x: x == 0
    else:
        return None

    h, w = data.shape
    if not (0 <= v < h and 0 <= u < w):
        return None

    val = data[v, u]
    if not invalid(val):
        return float(val) * scale

    # 像素无效 → 3x3 邻域中值回退
    r_min, r_max = max(0, v - 1), min(h, v + 2)
    c_min, c_max = max(0, u - 1), min(w, u + 2)
    patch = data[r_min:r_max, c_min:c_max]
    valid_mask = ~invalid(patch)
    if valid_mask.any():
        return float(np.median(patch[valid_mask])) * scale
    return None


def pixel_to_camera_3d(u, v, Z, camera_info):
    """
    针孔相机模型反投影: 像素坐标 + 深度 → 相机光学坐标系 3D 点。

    Args:
        u: float, 像素 x 坐标
        v: float, 像素 y 坐标
        Z: float, 深度值 (米)
        camera_info: sensor_msgs/CameraInfo, 内参矩阵 k = [fx,0,cx, 0,fy,cy, 0,0,1]

    Returns:
        tuple: (X, Y, Z) 在 camera_optical_frame 下, 单位米
    """
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return (X, Y, Z)


def transform_point(tf_buffer, x, y, z, source_frame, target_frame, timeout_sec=1.0):
    """
    使用 TF2 将点从 source_frame 变换到 target_frame。

    Args:
        tf_buffer: tf2_ros.Buffer 实例
        x, y, z: float, 源坐标系下的三维坐标 (米)
        source_frame: str, 源坐标系名
        target_frame: str, 目标坐标系名
        timeout_sec: float, 等待 TF 的超时秒数

    Returns:
        tuple: (x', y', z') 在 target_frame 下, 或 None (变换失败)
    """
    import logging
    import rclpy
    from geometry_msgs.msg import PointStamped

    _logger = logging.getLogger(__name__)

    point = PointStamped()
    point.header.frame_id = source_frame
    # Time() → sec=0,nsec=0 → "latest available transform" in TF2
    point.header.stamp = rclpy.time.Time().to_msg()
    point.point.x = x
    point.point.y = y
    point.point.z = z

    try:
        transformed = tf_buffer.transform(
            point, target_frame,
            timeout=rclpy.duration.Duration(seconds=timeout_sec)
        )
        return (transformed.point.x, transformed.point.y, transformed.point.z)
    except Exception as exc:
        _logger.warning(f"TF transform failed: {source_frame}→{target_frame}: {exc}")
        return None
