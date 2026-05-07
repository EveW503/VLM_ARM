"""
相机工具函数: 深度图查询、像素反投影、TF2 坐标变换。
"""
import numpy as np


def get_depth_at_pixel(depth_msg, u, v):
    """
    从深度图中查询指定像素的深度值 (米)。
    如果目标像素无效 (深度为 0), 回退到 3x3 邻域中值。

    Args:
        depth_msg: sensor_msgs/Image, encoding="16UC1", 单位为毫米
        u: int, 像素列坐标
        v: int, 像素行坐标

    Returns:
        float: 深度值 (米), 或 None (查询失败)
    """
    u = int(u)
    v = int(v)

    data = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
        depth_msg.height, depth_msg.width
    )
    h, w = data.shape

    if 0 <= v < h and 0 <= u < w:
        val = data[v, u]
        if val > 0:
            return val / 1000.0

    # 3x3 邻域中值回退
    r_min, r_max = max(0, v - 1), min(h, v + 2)
    c_min, c_max = max(0, u - 1), min(w, u + 2)
    patch = data[r_min:r_max, c_min:c_max]
    valid = patch[patch > 0]
    if len(valid) > 0:
        return float(np.median(valid)) / 1000.0
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
    import rclpy
    from geometry_msgs.msg import PointStamped

    point = PointStamped()
    point.header.frame_id = source_frame
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
    except Exception:
        return None
