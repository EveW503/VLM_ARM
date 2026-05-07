# VLM + ROS 2 抓取管线实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 VLM 两阶段推理驱动的机械臂抓取闭环——vlm_bridge（决策+映射层）和 grab_action（执行层），支撑单目标无遮挡场景的完整抓取演示。

**Architecture:** 4 个新文件 + 1 个重写 + 2 个修改。vlm_bridge 负责双相机切换、VLM 调用、2D→3D 映射并发布目标位姿；grab_action 负责状态机驱动的全流程抓取执行（MoveIt 规划 + LinkAttacher + 夹爪控制）。两个节点通过 `/target_pre_grasp`、`/target_pose`、`/grab_status` 三个 topic 通信。

**Tech Stack:** ROS 2 Humble, Python 3.10, MoveIt 2 (MoveGroup action), Qwen3 VL Plus (DashScope OpenAI 兼容接口), OpenCV, NumPy, TF2

---

## 文件映射

```
position_topic/position_topic/
├── vlm_bridge.py      ← 重写: 主节点（状态机、ROS 接口、双相机管理）
├── vlm_client.py      ← 新建: VLM API 调用封装（图像编码、API 调用、JSON 解析）
├── camera_utils.py    ← 新建: 深度查询、像素反投影、TF2 坐标变换
├── prompts.py         ← 新建: 阶段一/阶段二 System Prompt 常量
└── grab_action.py     ← 新建: 抓取全流程状态机

position_topic/
├── setup.py           ← 修改: 注册 grab_action entry_point
└── launch/move_demo.launch.py ← 修改: 添加 vlm_bridge + grab_action 节点
```

每个文件的职责边界：
- `prompts.py` — 纯字符串常量，无依赖
- `camera_utils.py` — 纯函数，依赖 NumPy + TF2 ROS 消息类型
- `vlm_client.py` — VLM API 调用类，依赖 `openai` + `cv_bridge` + `prompts.py`
- `vlm_bridge.py` — ROS 2 Node，组合 `VlmClient` + `camera_utils` 函数
- `grab_action.py` — ROS 2 Node，独立于 vlm_bridge，仅靠 topic 通信

---

### Task 1: 安装依赖

**Files:** None (environment setup)

- [ ] **Step 1: 安装 openai Python 包**

Run: `pip install openai`

Expected: `Successfully installed openai-<version>`

- [ ] **Step 2: 验证导入**

Run: `python3 -c "import openai; from cv_bridge import CvBridge; import numpy; print('OK')"`

Expected: `OK`

- [ ] **Step 3: 验证 DashScope API Key 环境变量可读**

Run: `python3 -c "import os; key=os.environ.get('DASHSCOPE_API_KEY'); print('SET' if key else 'MISSING')"`

Expected: `SET`

- [ ] **Step 4: 验证 API 连通性**

Run: `python3 -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
r = client.chat.completions.create(model='qwen3-vl-plus', messages=[{'role':'user','content':'Say hi in one word'}], max_tokens=10)
print(r.choices[0].message.content)
"`

Expected: 返回简短问候文本

- [ ] **Step 5: Commit**

```bash
# 不需要 commit，环境配置不属于代码仓库
```

---

### Task 2: 创建 prompts.py — VLM 提示词模板

**Files:**
- Create: `position_topic/position_topic/prompts.py`

- [ ] **Step 1: 创建文件**

```python
"""
VLM System Prompt 模板。
阶段一: 全局场景理解 + 目标粗定位 (Gemini 335 俯视)
阶段二: 近距抓取点精确定位 (ee_camera 手眼)
"""

STAGE1_SYSTEM_PROMPT = """\
你是一个农业采摘机器人的视觉系统。你的任务是分析俯视视角的草莓种植场景图像。

请完成以下任务:
1. 识别图像中所有可见的草莓果实
2. 挑选出最适合采摘的一颗（成熟度最高、最清晰可见）
3. 返回该草莓果实中心的像素坐标

请严格按以下 JSON 格式回复，不要包含任何其他文字:
{
  "target": {
    "pixel_x": <整数, 果实中心在图像中的x坐标(列)>,
    "pixel_y": <整数, 果实中心在图像中的y坐标(行)>,
    "label": "<描述, 如 'ripe_strawberry'>",
    "confidence": <0.0-1.0之间的浮点数, 置信度>
  }
}

注意:
- 图像分辨率 640x480, 左上角为 (0,0)
- 忽略画面中可能出现的机械臂部件
- 如果没有发现任何可采摘的草莓, pixel_x 和 pixel_y 均设为 -1
"""

STAGE2_SYSTEM_PROMPT = """\
你是一个农业采摘机器人的手眼视觉系统。你的任务是分析末端夹爪附近的近距离图像。

当前图像来自固定在机械臂末端的手眼相机, 已经非常靠近目标果实。

请完成以下任务:
1. 仔细观察图像中距离最近的草莓果实
2. 定位该草莓果柄(蒂部)的精确像素坐标——夹具应该夹住的位置
3. 如果图像中没有清晰的草莓果实, 返回 -1 坐标

请严格按以下 JSON 格式回复, 不要包含任何其他文字:
{
  "target": {
    "pixel_x": <整数, 果柄在图像中的x坐标(列)>,
    "pixel_y": <整数, 果柄在图像中的y坐标(行)>,
    "label": "<描述, 如 'strawberry_peduncle'>",
    "confidence": <0.0-1.0之间的浮点数>
  }
}

注意:
- 图像分辨率 640x480, 左上角为 (0,0)
- 果柄通常在红色果实的顶部, 与绿色萼片相连
- 如果没有发现清晰的草莓果实, pixel_x 和 pixel_y 均设为 -1
"""

DEFAULT_USER_INSTRUCTION = "请分析这张图像, 找出最适合采摘的草莓果实并返回其坐标。"
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import sys; sys.path.insert(0, 'src/position_topic'); from position_topic.prompts import STAGE1_SYSTEM_PROMPT; print('OK:', len(STAGE1_SYSTEM_PROMPT), 'chars')"`

Expected: `OK: <number> chars`

- [ ] **Step 3: Commit**

```bash
git add src/position_topic/position_topic/prompts.py
git commit -m "feat: add VLM system prompt templates for two-stage grasping"
```

---

### Task 3: 创建 camera_utils.py — 图像到 3D 坐标工具函数

**Files:**
- Create: `position_topic/position_topic/camera_utils.py`

- [ ] **Step 1: 创建文件**

```python
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
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import sys; sys.path.insert(0, 'src/position_topic'); from position_topic.camera_utils import get_depth_at_pixel, pixel_to_camera_3d, transform_point; print('OK')"`

Expected: `OK`

- [ ] **Step 3: 运行单元测试验证反投影**

Run: `python3 -c "
from sensor_msgs.msg import CameraInfo
from position_topic.camera_utils import pixel_to_camera_3d

info = CameraInfo()
info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

# 光心像素 → 应得 (0, 0, Z)
X, Y, Z = pixel_to_camera_3d(320.0, 240.0, 1.5, info)
assert abs(X) < 1e-6 and abs(Y) < 1e-6 and abs(Z - 1.5) < 1e-6, f'光心测试失败: {(X,Y,Z)}'

# 非光心像素
X, Y, Z = pixel_to_camera_3d(420.0, 340.0, 1.0, info)
assert abs(X - 0.2) < 1e-6, f'X 期望 0.2, 得到 {X}'
assert abs(Y - 0.2) < 1e-6, f'Y 期望 0.2, 得到 {Y}'
print('所有测试通过')
"`

Expected: `所有测试通过`

- [ ] **Step 4: Commit**

```bash
git add src/position_topic/position_topic/camera_utils.py
git commit -m "feat: add camera utils for depth lookup, unprojection, and TF2 transform"
```

---

### Task 4: 创建 vlm_client.py — VLM API 调用封装

**Files:**
- Create: `position_topic/position_topic/vlm_client.py`

- [ ] **Step 1: 创建文件**

```python
"""
VLM API 调用封装: 图像编码、DashScope API 调用、JSON 响应解析。
"""
import base64
import json
import re

import cv2
from cv_bridge import CvBridge
from openai import OpenAI


class VlmClient:
    """
    Qwen3 VL Plus 客户端, 通过 DashScope OpenAI 兼容接口调用。
    """

    def __init__(self, api_key=None, model="qwen3-vl-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self._model = model
        self._bridge = CvBridge()
        if api_key is None:
            import os
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def encode_image(self, image_msg):
        """
        将 ROS sensor_msgs/Image 编码为 base64 JPEG 字符串。

        Args:
            image_msg: sensor_msgs/Image (bgr8 或 rgb8)

        Returns:
            str: base64 编码的 JPEG 图像, 含 "data:image/jpeg;base64," 前缀
        """
        cv_image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        _, buffer = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def call_vlm(self, system_prompt, user_instruction, image_data_url, max_tokens=300):
        """
        调用 VLM 进行多模态推理。

        Args:
            system_prompt: str, 系统提示词
            user_instruction: str, 用户指令文本
            image_data_url: str, base64 编码后的图像 data URL
            max_tokens: int, 最大输出 token 数

        Returns:
            dict 或 None: 解析后的 JSON 响应, 格式:
                {"target": {"pixel_x": int, "pixel_y": int, "label": str, "confidence": float}}
            失败时返回 None
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": image_data_url}},
                            {"type": "text", "text": user_instruction},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            text = response.choices[0].message.content
            return self._parse_response(text)
        except Exception:
            return None

    def _parse_response(self, text):
        """
        从 VLM 文本响应中提取 JSON。

        Args:
            text: str, VLM 返回的原始文本

        Returns:
            dict 或 None: 解析后的 JSON
        """
        if text is None:
            return None
        text = text.strip()
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试从 markdown 代码块中提取
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试找到第一个 { 到最后一个 }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import sys; sys.path.insert(0, 'src/position_topic'); from position_topic.vlm_client import VlmClient; print('OK')"`

Expected: `OK`

- [ ] **Step 3: 验证 JSON 解析逻辑**

Run: `python3 -c "
from position_topic.vlm_client import VlmClient
c = VlmClient(api_key='fake')
# 测试纯 JSON
r = c._parse_response('{\"target\":{\"pixel_x\":320,\"pixel_y\":240,\"label\":\"test\",\"confidence\":0.9}}')
assert r['target']['pixel_x'] == 320
# 测试 markdown 包裹
r = c._parse_response('```json\n{\"target\":{\"pixel_x\":100,\"pixel_y\":200,\"label\":\"x\",\"confidence\":0.5}}\n```')
assert r['target']['pixel_x'] == 100
# 测试前后有文本
r = c._parse_response('这里是分析: {\"target\":{\"pixel_x\":50,\"pixel_y\":60,\"label\":\"y\",\"confidence\":0.8}}更多文本')
assert r['target']['pixel_x'] == 50
print('所有解析测试通过')
"`

Expected: `所有解析测试通过`

- [ ] **Step 4: Commit**

```bash
git add src/position_topic/position_topic/vlm_client.py
git commit -m "feat: add VLM client wrapper with DashScope OpenAI-compatible API"
```

---

### Task 5: 重写 vlm_bridge.py — 主推理节点

**Files:**
- Modify: `position_topic/position_topic/vlm_bridge.py` (重写)

- [ ] **Step 1: 重写完整节点**

```python
"""
VLM Bridge 节点: 双阶段 VLM 推理 + 2D→3D 映射 + 目标位姿发布。

阶段一: Gemini 335 全局推理 → 粗定位 → /target_pre_grasp
阶段二: ee_camera 近距精定位 → /target_pose
"""
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
import tf2_ros

from .camera_utils import get_depth_at_pixel, pixel_to_camera_3d, transform_point
from .prompts import (
    STAGE1_SYSTEM_PROMPT, STAGE2_SYSTEM_PROMPT, DEFAULT_USER_INSTRUCTION,
)
from .vlm_client import VlmClient


class VlmBridge(Node):
    STAGE_IDLE = "idle"
    STAGE1_QUERY = "stage1_query"
    STAGE1_WAIT = "stage1_wait"
    STAGE2_QUERY = "stage2_query"

    def __init__(self):
        super().__init__("vlm_bridge")

        # --- 参数 ---
        self.declare_parameter("dry_run", False)
        self._dry_run = self.get_parameter("dry_run").value

        # --- VLM 客户端 ---
        self._vlm = VlmClient()

        # --- TF2 ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Gemini 335 相机订阅 ---
        self._gemini_rgb = None
        self._gemini_depth = None
        self._gemini_info = None

        self.create_subscription(
            Image, "/camera/color/image_raw", self._gemini_rgb_cb, 10
        )
        self.create_subscription(
            Image, "/camera/depth/image_raw", self._gemini_depth_cb, 10
        )
        self.create_subscription(
            CameraInfo, "/camera/depth/camera_info", self._gemini_info_cb, 10
        )

        # --- ee_camera 手眼相机订阅 ---
        self._ee_rgb = None
        self._ee_depth = None
        self._ee_info = None

        self.create_subscription(
            Image, "/so101/camera/image_raw", self._ee_rgb_cb, 10
        )
        self.create_subscription(
            Image, "/so101/camera/depth/image_raw", self._ee_depth_cb, 10
        )
        self.create_subscription(
            CameraInfo, "/so101/camera/depth/camera_info", self._ee_info_cb, 10
        )

        # --- 任务指令 ---
        self._task_instruction = DEFAULT_USER_INSTRUCTION
        self.create_subscription(
            String, "/task_command", self._task_cmd_cb, 10
        )

        # --- 抓取状态反馈 ---
        self.create_subscription(
            String, "/grab_status", self._grab_status_cb, 10
        )

        # --- 输出 ---
        self._pre_grasp_pub = self.create_publisher(
            PoseStamped, "/target_pre_grasp", 10
        )
        self._target_pub = self.create_publisher(
            PoseStamped, "/target_pose", 10
        )

        # --- 状态机 ---
        self._state = self.STAGE_IDLE
        self._vlm_in_progress = False
        self._state_timer = self.create_timer(0.1, self._state_machine_tick)

        self.get_logger().info("VLM Bridge 节点已启动")

    # ── 回调: Gemini 335 ──────────────────────────────

    def _gemini_rgb_cb(self, msg):
        self._gemini_rgb = msg

    def _gemini_depth_cb(self, msg):
        self._gemini_depth = msg

    def _gemini_info_cb(self, msg):
        if self._gemini_info is None:
            self.get_logger().info(
                f"Gemini335 内参: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}"
            )
        self._gemini_info = msg

    # ── 回调: ee_camera ────────────────────────────────

    def _ee_rgb_cb(self, msg):
        self._ee_rgb = msg

    def _ee_depth_cb(self, msg):
        self._ee_depth = msg

    def _ee_info_cb(self, msg):
        if self._ee_info is None:
            self.get_logger().info(
                f"ee_camera 内参: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}"
            )
        self._ee_info = msg

    # ── 回调: 指令 / 状态 ──────────────────────────────

    def _task_cmd_cb(self, msg):
        self._task_instruction = msg.data
        self.get_logger().info(f"收到任务指令: {msg.data}")
        if self._state == self.STAGE_IDLE:
            self.get_logger().info("触发阶段一推理")
            self._state = self.STAGE1_QUERY

    def _grab_status_cb(self, msg):
        if msg.data == "stage1_done" and self._state == self.STAGE1_WAIT:
            self.get_logger().info("收到 stage1_done, 触发阶段二推理")
            self._state = self.STAGE2_QUERY

    # ── 状态机 ─────────────────────────────────────────

    def _state_machine_tick(self):
        if self._vlm_in_progress:
            return

        if self._state == self.STAGE1_QUERY:
            if self._gemini_rgb is None:
                self.get_logger().warn("等待 Gemini335 RGB 图像...")
                return
            if self._gemini_depth is None:
                self.get_logger().warn("等待 Gemini335 深度图...")
                return
            if self._gemini_info is None:
                self.get_logger().warn("等待 Gemini335 camera_info...")
                return
            self._vlm_in_progress = True
            threading.Thread(target=self._run_stage1).start()

        elif self._state == self.STAGE2_QUERY:
            if self._ee_rgb is None:
                self.get_logger().warn("等待 ee_camera RGB 图像...")
                return
            if self._ee_depth is None:
                self.get_logger().warn("等待 ee_camera 深度图...")
                return
            if self._ee_info is None:
                self.get_logger().warn("等待 ee_camera camera_info...")
                return
            self._vlm_in_progress = True
            threading.Thread(target=self._run_stage2).start()

    # ── 阶段一: 全局推理 ───────────────────────────────

    def _run_stage1(self):
        self.get_logger().info("阶段一: 调用 VLM (Gemini335)...")
        rgb = self._gemini_rgb
        depth = self._gemini_depth
        info = self._gemini_info

        img_b64 = self._vlm.encode_image(rgb)
        result = self._vlm.call_vlm(
            STAGE1_SYSTEM_PROMPT, self._task_instruction, img_b64
        )
        if result is None:
            self.get_logger().error("阶段一 VLM 调用失败")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        target = result.get("target", {})
        px = target.get("pixel_x", -1)
        py = target.get("pixel_y", -1)
        confidence = target.get("confidence", 0.0)
        label = target.get("label", "unknown")

        if px < 0 or py < 0:
            self.get_logger().warn("VLM 未发现可采摘目标")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        self.get_logger().info(
            f"阶段一结果: ({px}, {py}) label={label} conf={confidence:.2f}"
        )

        Z = get_depth_at_pixel(depth, px, py)
        if Z is None:
            self.get_logger().error("阶段一: 深度查询失败")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        X_cam, Y_cam, Z_cam = pixel_to_camera_3d(px, py, Z, info)
        pt = transform_point(
            self._tf_buffer,
            X_cam, Y_cam, Z_cam,
            "camera_depth_optical_frame", "base",
        )
        if pt is None:
            self.get_logger().error("阶段一: TF 变换失败 (camera_depth_optical_frame→base)")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        self.get_logger().info(
            f"阶段一 3D (base): ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
        )

        if not self._dry_run:
            msg = PoseStamped()
            msg.header.frame_id = "base"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position = Point(x=pt[0], y=pt[1], z=pt[2])
            msg.pose.orientation.w = 1.0
            self._pre_grasp_pub.publish(msg)
            self.get_logger().info("已发布 /target_pre_grasp")

        self._state = self.STAGE1_WAIT
        self._vlm_in_progress = False

    # ── 阶段二: 精确定位 ───────────────────────────────

    def _run_stage2(self):
        self.get_logger().info("阶段二: 调用 VLM (ee_camera)...")
        rgb = self._ee_rgb
        depth = self._ee_depth
        info = self._ee_info

        img_b64 = self._vlm.encode_image(rgb)
        result = self._vlm.call_vlm(
            STAGE2_SYSTEM_PROMPT, self._task_instruction, img_b64
        )
        if result is None:
            self.get_logger().error("阶段二 VLM 调用失败")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        target = result.get("target", {})
        px = target.get("pixel_x", -1)
        py = target.get("pixel_y", -1)
        confidence = target.get("confidence", 0.0)
        label = target.get("label", "unknown")

        if px < 0 or py < 0:
            self.get_logger().warn("VLM 阶段二未发现目标")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        self.get_logger().info(
            f"阶段二结果: ({px}, {py}) label={label} conf={confidence:.2f}"
        )

        Z = get_depth_at_pixel(depth, px, py)
        if Z is None:
            self.get_logger().error("阶段二: 深度查询失败")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        X_cam, Y_cam, Z_cam = pixel_to_camera_3d(px, py, Z, info)
        pt = transform_point(
            self._tf_buffer,
            X_cam, Y_cam, Z_cam,
            "ee_camera_optical_link", "base",
        )
        if pt is None:
            self.get_logger().error("阶段二: TF 变换失败 (ee_camera_optical_link→base)")
            self._state = self.STAGE_IDLE
            self._vlm_in_progress = False
            return

        self.get_logger().info(
            f"阶段二 3D (base): ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
        )

        if not self._dry_run:
            msg = PoseStamped()
            msg.header.frame_id = "base"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position = Point(x=pt[0], y=pt[1], z=pt[2])
            msg.pose.orientation.w = 1.0
            self._target_pub.publish(msg)
            self.get_logger().info("已发布 /target_pose")

        self._state = self.STAGE_IDLE
        self._vlm_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = VlmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import py_compile; py_compile.compile('src/position_topic/position_topic/vlm_bridge.py', doraise=True); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/position_topic/position_topic/vlm_bridge.py
git commit -m "feat: rewrite vlm_bridge with two-stage state machine and dual-camera support"
```

---

### Task 6: 创建 grab_action.py — 抓取执行节点

**Files:**
- Create: `position_topic/position_topic/grab_action.py`

- [ ] **Step 1: 创建文件**

```python
"""
Grab Action 节点: 抓取全流程状态机。

订阅 /target_pre_grasp (阶段一粗定位) 和 /target_pose (阶段二精定位),
执行: 靠近→下移→夹取→抬起→搬运→放置→释放。
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume, Constraints, MotionPlanRequest,
    PlanningOptions, PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from linkattacher_msgs.srv import AttachLink, DetachLink
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GrabAction(Node):
    # 状态常量
    IDLE = "idle"
    PRE_GRASP_PLAN = "pre_grasp_plan"
    PRE_GRASP_EXEC = "pre_grasp_exec"
    AWAIT_STAGE2 = "await_stage2"
    APPROACH_PLAN = "approach_plan"
    APPROACH_EXEC = "approach_exec"
    GRASP = "grasp"
    LIFT_PLAN = "lift_plan"
    LIFT_EXEC = "lift_exec"
    TRANSPORT_PLAN = "transport_plan"
    TRANSPORT_EXEC = "transport_exec"
    PLACE = "place"
    FAILED = "failed"

    def __init__(self):
        super().__init__("grab_action")

        # --- 参数 ---
        self.declare_parameter("placement_position", [0.2, -0.15, 0.25])
        self.declare_parameter("pre_grasp_z_offset", 0.08)
        self.declare_parameter("approach_step", 0.01)
        self.declare_parameter("gripper_close_pos", -0.17)
        self.declare_parameter("gripper_open_pos", 0.5)
        self.declare_parameter("stage2_timeout", 10.0)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.3)

        # --- 输入订阅 ---
        self.create_subscription(
            PoseStamped, "/target_pre_grasp", self._pre_grasp_cb, 10
        )
        self.create_subscription(
            PoseStamped, "/target_pose", self._target_pose_cb, 10
        )

        # --- 状态反馈 ---
        self._status_pub = self.create_publisher(String, "/grab_status", 10)

        # --- MoveGroup action client ---
        self._move_action = ActionClient(self, MoveGroup, "/move_action")
        if not self._move_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("无法连接 /move_action")

        # --- LinkAttacher service clients ---
        self._attach_cli = self.create_client(AttachLink, "/ATTACHLINK")
        self._detach_cli = self.create_client(DetachLink, "/DETACHLINK")

        # --- Gripper action client ---
        self._gripper_action = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

        # --- 状态机 ---
        self._state = self.IDLE
        self._state_before_await = None
        self._pre_grasp_target = None  # 预抓取位姿
        self._grasp_target = None      # 精确抓取位姿
        self._current_pose = None      # 当前末端位姿
        self._active_goal_handle = None

        self.get_logger().info("Grab Action 节点已启动")

    # ── 回调 ──────────────────────────────────────────

    def _pre_grasp_cb(self, msg):
        if self._state == self.IDLE:
            self.get_logger().info(
                f"收到预抓取目标: ({msg.pose.position.x:.3f}, "
                f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
            )
            self._pre_grasp_target = msg
            self._grasp_target = None
            self._transition(self.PRE_GRASP_PLAN)

    def _target_pose_cb(self, msg):
        self.get_logger().info(
            f"收到精确抓取位姿: ({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )
        self._grasp_target = msg
        if self._state == self.AWAIT_STAGE2:
            self._transition(self.APPROACH_PLAN)

    # ── 状态机驱动 ────────────────────────────────────

    def _transition(self, new_state):
        self._state = new_state
        if new_state == self.PRE_GRASP_PLAN:
            self._plan_pre_grasp()
        elif new_state == self.APPROACH_PLAN:
            self._plan_approach()
        elif new_state == self.LIFT_PLAN:
            self._plan_lift()
        elif new_state == self.TRANSPORT_PLAN:
            self._plan_transport()
        elif new_state == self.GRASP:
            self._do_grasp()
        elif new_state == self.PLACE:
            self._do_place()
        elif new_state == self.FAILED:
            self.get_logger().error("抓取流程失败")
            self._publish_status("error")
            self._state = self.IDLE

    def _publish_status(self, status):
        msg = String(data=status)
        self._status_pub.publish(msg)

    # ── 预抓取 ──────────────────────────────────────

    def _plan_pre_grasp(self):
        self.get_logger().info("规划预抓取位姿...")
        target = self._pre_grasp_target
        if target is None:
            self._transition(self.FAILED)
            return

        # 将目标 Z 上移 pre_grasp_z_offset 作为预抓取点
        pre_grasp_pose = Pose()
        pre_grasp_pose.position = Point(
            x=target.pose.position.x,
            y=target.pose.position.y,
            z=target.pose.position.z + self.get_parameter("pre_grasp_z_offset").value,
        )
        pre_grasp_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            pre_grasp_pose, "base",
            success_callback=self._on_pre_grasp_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_pre_grasp_done(self):
        self.get_logger().info("预抓取完成, 等待阶段二精定位...")
        self._publish_status("stage1_done")
        self._stage2_timer = self.create_timer(
            self.get_parameter("stage2_timeout").value,
            self._stage2_timeout_cb,
        )
        self._state = self.AWAIT_STAGE2

    def _stage2_timeout_cb(self):
        if self._state != self.AWAIT_STAGE2:
            return
        self.get_logger().warn("阶段二超时, 使用降级策略")
        self.destroy_timer(self._stage2_timer)
        # 降级: 用 pre_grasp 位姿减去 Z 偏移作为抓取位姿
        if self._grasp_target is None:
            fallback = Pose()
            t = self._pre_grasp_target
            fb_z = t.pose.position.z - self.get_parameter("pre_grasp_z_offset").value
            fallback.position = Point(x=t.pose.position.x, y=t.pose.position.y, z=fb_z)
            fallback.orientation = Quaternion(w=1.0)
            self._grasp_target = PoseStamped()
            self._grasp_target.header.frame_id = "base"
            self._grasp_target.pose = fallback
        self._transition(self.APPROACH_PLAN)

    # ── 接近 (Cartesian 直线下移) ────────────────────

    def _plan_approach(self):
        self.get_logger().info("规划接近路径...")
        if self._grasp_target is None:
            self._transition(self.FAILED)
            return

        self._send_move_request(
            self._grasp_target.pose, self._grasp_target.header.frame_id or "base",
            success_callback=self._on_approach_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_approach_done(self):
        self.get_logger().info("到达抓取位姿")
        self._transition(self.GRASP)

    # ── 抓取 ──────────────────────────────────────

    def _do_grasp(self):
        self.get_logger().info("闭合夹爪...")
        self._send_gripper_command(
            self.get_parameter("gripper_close_pos").value,
            done_callback=self._on_gripper_close_done,
            fail_callback=lambda: self._transition(self.FAILED),
        )

    def _on_gripper_close_done(self):
        self.get_logger().info("调用 AttachLink...")
        req = AttachLink.Request()
        req.model1_name = "so101"
        req.link1_name = "gripper"
        req.model2_name = "target_box"
        req.link2_name = "box_link"

        future = self._attach_cli.call_async(req)
        future.add_done_callback(self._on_attach_done)

    def _on_attach_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info("物体已附着")
                self._publish_status("stage2_done")
                self._transition(self.LIFT_PLAN)
            else:
                self.get_logger().error(f"AttachLink 失败: {resp.message}")
                self._transition(self.FAILED)
        except Exception:
            self.get_logger().error("AttachLink 调用异常")
            self._transition(self.FAILED)

    # ── 抬起 ──────────────────────────────────────

    def _plan_lift(self):
        self.get_logger().info("规划抬起路径...")
        lift_pose = Pose()
        lift_pose.position = Point(
            x=self._pre_grasp_target.pose.position.x,
            y=self._pre_grasp_target.pose.position.y,
            z=self._pre_grasp_target.pose.position.z
            + self.get_parameter("pre_grasp_z_offset").value,
        )
        lift_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            lift_pose, "base",
            success_callback=lambda: self._transition(self.TRANSPORT_PLAN),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 搬运 ──────────────────────────────────────

    def _plan_transport(self):
        self.get_logger().info("规划搬运到放置位姿...")
        placement = self.get_parameter("placement_position").value
        place_pose = Pose()
        place_pose.position = Point(
            x=placement[0], y=placement[1], z=placement[2]
        )
        place_pose.orientation = Quaternion(w=1.0)

        self._send_move_request(
            place_pose, "base",
            success_callback=lambda: self._transition(self.PLACE),
            fail_callback=lambda: self._transition(self.FAILED),
        )

    # ── 放置 ──────────────────────────────────────

    def _do_place(self):
        self.get_logger().info("释放物体...")
        req = DetachLink.Request()
        req.model1_name = "so101"
        req.link1_name = "gripper"
        req.model2_name = "target_box"
        req.link2_name = "box_link"

        future = self._detach_cli.call_async(req)
        future.add_done_callback(self._on_detach_done)

    def _on_detach_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info("物体已释放")
            else:
                self.get_logger().warn(f"DetachLink: {resp.message}")
        except Exception:
            self.get_logger().warn("DetachLink 调用异常")

        # 张开夹爪
        self._send_gripper_command(
            self.get_parameter("gripper_open_pos").value,
            done_callback=self._on_place_complete,
            fail_callback=self._on_place_complete,
        )

    def _on_place_complete(self):
        self.get_logger().info("抓取流程完成!")
        self._publish_status("grasp_complete")
        self._state = self.IDLE

    # ── MoveGroup 封装 ────────────────────────────

    def _send_move_request(self, target_pose, frame_id, *,
                           success_callback, fail_callback):
        goal = MoveGroup.Goal()
        goal.request = self._build_request(target_pose, frame_id)
        goal.planning_options = PlanningOptions()

        send_future = self._move_action.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._move_response_cb(f, success_callback, fail_callback)
        )

    def _build_request(self, pose, frame_id):
        req = MotionPlanRequest()
        req.group_name = "arm"
        req.num_planning_attempts = 10
        req.allowed_planning_time = self.get_parameter("planning_time").value
        req.max_velocity_scaling_factor = self.get_parameter("velocity_scaling").value
        req.max_acceleration_scaling_factor = self.get_parameter("velocity_scaling").value

        constraints = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = "gripper"
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]

        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(pose)
        pc.constraint_region = region
        constraints.position_constraints.append(pc)
        req.goal_constraints.append(constraints)
        return req

    def _move_response_cb(self, future, success_cb, fail_cb):
        try:
            goal_handle = future.result()
        except Exception:
            self.get_logger().error("发送 MoveGroup 目标失败")
            fail_cb()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("MoveGroup 拒绝目标")
            fail_cb()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._move_result_cb(f, success_cb, fail_cb)
        )

    def _move_result_cb(self, future, success_cb, fail_cb):
        try:
            wrapped = future.result()
        except Exception:
            self.get_logger().error("获取 MoveGroup 结果失败")
            fail_cb()
            return

        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"MoveGroup 成功, time={wrapped.result.planning_time:.2f}s")
            success_cb()
        else:
            self.get_logger().warn(
                f"MoveGroup 失败, status={wrapped.status}, "
                f"error={wrapped.result.error_code.val}"
            )
            fail_cb()

    # ── 夹爪控制 ──────────────────────────────────

    def _send_gripper_command(self, position, *, done_callback, fail_callback):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["6"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start.sec = 1
        goal.trajectory.points = [pt]

        send_future = self._gripper_action.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._gripper_response_cb(f, done_callback, fail_callback)
        )

    def _gripper_response_cb(self, future, done_cb, fail_cb):
        try:
            goal_handle = future.result()
        except Exception:
            self.get_logger().error("发送夹爪指令失败")
            fail_cb()
            return
        if not goal_handle.accepted:
            self.get_logger().warn("夹爪控制器拒绝指令")
            fail_cb()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._gripper_result_cb(f, done_cb, fail_cb)
        )

    def _gripper_result_cb(self, future, done_cb, fail_cb):
        try:
            wrapped = future.result()
        except Exception:
            fail_cb()
            return
        if wrapped.result.error_code == 0:
            done_cb()
        else:
            self.get_logger().warn(f"夹爪执行失败: {wrapped.result.error_string}")
            fail_cb()


def main(args=None):
    rclpy.init(args=args)
    node = GrabAction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /home/ljq/lerobot_ws && python3 -c "import py_compile; py_compile.compile('src/position_topic/position_topic/grab_action.py', doraise=True); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/position_topic/position_topic/grab_action.py
git commit -m "feat: add grab_action node with full grasp pipeline state machine"
```

---

### Task 7: 修改 setup.py — 注册新 entry_point

**Files:**
- Modify: `position_topic/setup.py` (改 entry_points 的 console_scripts)

- [ ] **Step 1: 添加 grab_action entry_point**

Read the current `setup.py` first.

当前 `entry_points` 部分：
```python
entry_points={
    'console_scripts': [
        'position_publisher = position_topic.position_publisher:main',
        'position_subscriber = position_topic.position_subscriber:main',
        'vlm_bridge = position_topic.vlm_bridge:main',
    ],
},
```

修改为：
```python
entry_points={
    'console_scripts': [
        'position_publisher = position_topic.position_publisher:main',
        'position_subscriber = position_topic.position_subscriber:main',
        'vlm_bridge = position_topic.vlm_bridge:main',
        'grab_action = position_topic.grab_action:main',
    ],
},
```

- [ ] **Step 2: 重新构建 position_topic 包**

Run: `cd /home/ljq/lerobot_ws && colcon build --symlink-install --packages-select position_topic`

Expected: `Summary: 1 package finished`

- [ ] **Step 3: 验证 entry_point 生效**

Run: `source /home/ljq/lerobot_ws/install/setup.bash && ros2 run position_topic grab_action --help 2>&1 || true`

Expected: 无 import error（节点可能会因为无 ROS 环境而退出，但不应有 Python import 错误）

- [ ] **Step 4: Commit**

```bash
git add src/position_topic/setup.py
git commit -m "feat: register grab_action entry_point in setup.py"
```

---

### Task 8: 修改 move_demo.launch.py — 集成新节点

**Files:**
- Modify: `position_topic/launch/move_demo.launch.py`

- [ ] **Step 1: 添加 vlm_bridge 和 grab_action 节点到 launch 文件**

当前 launch 文件中第 36-43 行定义了 `position_subscriber` 节点。在它后面添加 vlm_bridge 和 grab_action。

在 `position_subscriber` 定义之后，添加：

```python
    vlm_bridge = Node(
        package='position_topic',
        executable='vlm_bridge',
        name='vlm_bridge',
        output='screen'
    )

    grab_action = Node(
        package='position_topic',
        executable='grab_action',
        name='grab_action',
        output='screen',
        parameters=[{
            'placement_position': [0.2, -0.15, 0.25],
            'pre_grasp_z_offset': 0.08,
            'velocity_scaling': 0.3,
        }]
    )
```

并在 `return LaunchDescription([...])` 中添加这两个节点：

```python
    return LaunchDescription([
        gazebo_launch,
        controller_launch,
        moveit_launch,
        position_subscriber,
        vlm_bridge,
        grab_action,
    ])
```

- [ ] **Step 2: 重新构建**

Run: `cd /home/ljq/lerobot_ws && colcon build --symlink-install --packages-select position_topic`

Expected: `Summary: 1 package finished`

- [ ] **Step 3: Commit**

```bash
git add src/position_topic/launch/move_demo.launch.py
git commit -m "feat: integrate vlm_bridge and grab_action into move_demo launch"
```

---

### Task 9: 集成测试 — 端到端验证

**Files:** None (测试验证)

- [ ] **Step 1: 启动完整仿真环境**

Run: `source /home/ljq/lerobot_ws/install/setup.bash && ros2 launch position_topic move_demo.launch.py`

Expected: Gazebo + MoveIt + RViz 全部启动, vlm_bridge 和 grab_action 节点日志输出正常, 无 import/topic 错误

- [ ] **Step 2: 验证 vlm_bridge 相机订阅正常**

检查日志中应出现类似:
```
[VLM Bridge] Gemini335 内参: fx=...
[VLM Bridge] ee_camera 内参: fx=...
```

- [ ] **Step 3: 测试 dry_run 模式**

首先终止当前仿真（如果正在运行），然后用 dry_run 重新启动 vlm_bridge：

Run: `ros2 run position_topic vlm_bridge --ros-args -p dry_run:=true`

然后在另一个终端发布测试指令:
```bash
ros2 topic pub /task_command std_msgs/String "data: '摘一个草莓'" -1
```

Expected: vlm_bridge 调用 VLM，输出检测结果，但不发布 `/target_pre_grasp`

- [ ] **Step 4: 测试硬编码抓取全流程**

手动发布预抓取位姿，跳过 VLM 阶段一:
```bash
ros2 topic pub /target_pre_grasp geometry_msgs/PoseStamped "{header: {frame_id: 'base'}, pose: {position: {x: 0.25, y: 0.05, z: 0.28}, orientation: {w: 1.0}}}" -1
```

Expected: grab_action 规划/执行预抓取, 发布 "stage1_done", 等待 /target_pose

手动发布精确抓取位姿:
```bash
ros2 topic pub /target_pose geometry_msgs/PoseStamped "{header: {frame_id: 'base'}, pose: {position: {x: 0.25, y: 0.05, z: 0.17}, orientation: {w: 1.0}}}" -1
```

Expected: grab_action 执行下移→夹取→抬起→搬运→放置, 发布 "grasp_complete"

- [ ] **Step 5: 验证 LinkAttacher 调用**

在 Step 4 执行后检查 Gazebo 中 target_box 是否跟随 gripper 移动并最终放在指定位置。

- [ ] **Step 6: 测试端到端 VLM 管线**

启动完整仿真后, 发布任务指令:
```bash
ros2 topic pub /task_command std_msgs/String "data: '请摘一个草莓'" -1
```

Expected: vlm_bridge 两阶段推理 → grab_action 自动抓取全流程。

---

## 构建验证

全部 Task 完成后运行一次完整构建:

```bash
cd /home/ljq/lerobot_ws
colcon build --symlink-install --packages-select position_topic lerobot_description lerobot_controller lerobot_moveit linkattacher_msgs ros2_linkattacher
```

预期: 所有包构建成功, 无错误。
