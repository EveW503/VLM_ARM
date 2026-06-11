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


def get_min_depth_in_region(depth_msg, x1, y1, x2, y2):
    """
    取图像矩形区域内的最小有效深度值 (米)。
    用于从 VLM bbox 区域确定最近点——排除远处背景的干扰。

    Args:
        depth_msg: sensor_msgs/Image, encoding "16UC1" 或 "32FC1"
        x1, y1: 左上角像素坐标
        x2, y2: 右下角像素坐标

    Returns:
        float: 区域最小深度值 (米), 或 None
    """
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)

    enc = depth_msg.encoding
    if enc == "32FC1":
        data = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width)
        scale = 1.0
        invalid = np.isnan
    elif enc == "16UC1":
        data = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
            depth_msg.height, depth_msg.width)
        scale = 1.0 / 1000.0
        invalid = lambda x: x == 0
    else:
        return None

    h, w = data.shape
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)

    if x1 >= x2 or y1 >= y2:
        return None

    region = data[y1:y2, x1:x2]
    valid = region[~invalid(region)]
    if len(valid) > 0:
        return float(np.min(valid)) * scale
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
    import tf2_geometry_msgs  # noqa: F401 — 注册 geometry_msgs 的 TF2 类型支持
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


def compute_lookat_quaternion(p_obs, p_target):
    """
    计算从观察点 p_obs 指向目标点 p_target 的四元数。
    gripper X 轴 (= ee_camera optical Z) 对准目标方向。

    Args:
        p_obs: (3,) array-like, 观察点坐标 [x, y, z] (米)
        p_target: (3,) array-like, 目标点坐标 [x, y, z] (米)

    Returns:
        tuple: (qx, qy, qz, qw) 四元数, 或 None
    """
    from scipy.spatial.transform import Rotation

    p_obs = np.asarray(p_obs, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)

    direction = p_target - p_obs
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    x_axis = direction / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x_axis, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])

    y_axis = np.cross(world_up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    r = Rotation.from_matrix(R)
    q = r.as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
