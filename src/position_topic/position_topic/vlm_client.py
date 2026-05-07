"""
VLM API 调用封装: 图像编码、DashScope API 调用、JSON 响应解析。
"""
import base64
import json
import logging
import os
import re

import cv2
from cv_bridge import CvBridge
from openai import OpenAI

_logger = logging.getLogger(__name__)


class VlmClient:
    """
    Qwen3 VL Plus 客户端, 通过 DashScope OpenAI 兼容接口调用。
    """

    def __init__(self, api_key=None, model="qwen3-vl-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self._model = model
        self._bridge = CvBridge()
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            _logger.warning("DASHSCOPE_API_KEY 未设置, VLM 调用将失败")
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
                timeout=30.0,
            )
            text = response.choices[0].message.content
            return self._parse_response(text)
        except Exception as exc:
            _logger.error(f"VLM API 调用失败: {exc}", exc_info=True)
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
