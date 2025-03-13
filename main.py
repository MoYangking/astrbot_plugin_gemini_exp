from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, EventMessageType
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import *
import asyncio
import sys
import importlib
import requests
from io import BytesIO
import time
import tempfile
import os

@register("astrbot_plugin_geminiexp", "YourName", "基于 Google Gemini 2.0 Flash Experimental 多模态模型的插件", "1.0.0")
class GeminiExpPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.api_key = config.get("api_key", "")
        self.waiting_users = {}  # 存储正在等待输入的用户 {user_id: expiry_time}
        self.temp_dir = tempfile.mkdtemp(prefix="gemini_exp_")
        
        # 检查并安装 google-generativeai
        if not self._check_google_genai():
            self._install_google_genai()
        
        # 导入必要的模块
        global genai, types, PILImage
        from google import genai
        from google.genai import types
        from PIL import Image as PILImage
        
    def _check_google_genai(self) -> bool:
        """检查是否安装了 google-generativeai"""
        try:
            importlib.import_module('google.generativeai')
            return True
        except ImportError:
            return False

    def _install_google_genai(self):
        """安装 google-generativeai 包"""
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "google-generativeai", "pillow"])
            print("成功安装 google-generativeai 和 pillow 包")
        except subprocess.CalledProcessError as e:
            print(f"安装包失败: {str(e)}")
            raise
        
    @filter.command("祥 图片生成")
    async def gemini_exp(self, event: AstrMessageEvent):
        '''使用Gemini 2.0 Flash Experimental模型进行多模态交互'''
        # 检查API密钥是否配置
        if not self.api_key:
            yield event.plain_result("请联系管理员配置Gemini API密钥")
            return
        
        # 获取用户ID
        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        
        # 设置等待状态，有效期30秒
        expiry_time = time.time() + 30
        self.waiting_users[user_id] = expiry_time
        
        yield event.plain_result(f"好的 {user_name}，请在30秒内发送您的文本描述和图片（如有），系统将使用Gemini 2.0 Flash Experimental模型处理您的请求。")
    
    @filter.event_message_type(EventMessageType.ALL)
    async def handle_follow_up(self, event: AstrMessageEvent):
        """处理用户的后续消息"""
        user_id = event.get_sender_id()
        
        # 检查用户是否在等待状态
        if user_id not in self.waiting_users:
            return
        
        # 检查等待是否过期
        if time.time() > self.waiting_users[user_id]:
            del self.waiting_users[user_id]
            return
        
        # 移除用户的等待状态
        del self.waiting_users[user_id]
        
        # 获取消息内容
        message_chain = event.get_messages()
        text_content = event.message_str
        image_list = []
        
        # 从消息链中提取图片
        for msg in message_chain:
            if isinstance(msg, Image):
                try:
                    # 尝试获取图片URL
                    img_url = None
                    if hasattr(msg, 'url') and msg.url:
                        img_url = msg.url
                    
                    if img_url:
                        # 下载图片
                        response = requests.get(img_url, timeout=10)
                        if response.status_code == 200:
                            # 保存为临时文件
                            temp_img_path = os.path.join(self.temp_dir, f"input_{time.time()}.jpg")
                            with open(temp_img_path, 'wb') as f:
                                f.write(response.content)
                            
                            # 使用PIL打开图片
                            img = PILImage.open(temp_img_path)
                            image_list.append(img)
                except Exception as e:
                    logger.error(f"处理图片时出错: {str(e)}")
        
        if not text_content and not image_list:
            yield event.plain_result("请提供文本描述或图片。")
            return
        
        # 发送处理中的消息
        yield event.plain_result("正在处理您的请求，请稍候...")
        
        # 调用Gemini API
        try:
            result = await self.process_with_gemini(text_content, image_list)
            text_response = result.get('text', '无文本回复')
            image_paths = result.get('image_paths', [])
            
            # 构建回复消息链
            chain = [Plain(text_response)]
            
            # 添加图片到消息链
            for img_path in image_paths:
                chain.append(Image.fromFileSystem(img_path))
            
            yield event.chain_result(chain)
            
        except Exception as e:
            logger.error(f"Gemini API调用失败: {str(e)}")
            yield event.plain_result(f"处理失败: {str(e)}")
    
    async def process_with_gemini(self, text, images):
        """处理图片和文本，调用Gemini API"""
        try:
            # 初始化Gemini客户端
            client = genai.Client(api_key=self.api_key)
            
            # 准备内容
            contents = []
            if text:
                contents.append(text)
            
            # 添加图片
            for img in images:
                contents.append(img)
            
            if not contents:
                raise ValueError("没有有效的内容可以发送给Gemini API")
            
            # 调用API
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
            )

            
            # 解析响应
            result = {'text': '', 'image_paths': []}
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    result['text'] += part.text
                elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
                    # 将图片数据保存为临时文件
                    img_data = part.inline_data.data
                    img = PILImage.open(BytesIO(img_data))
                    
                    # 创建临时文件路径
                    temp_file_path = os.path.join(self.temp_dir, f"gemini_result_{time.time()}.png")
                    
                    # 保存图片
                    img.save(temp_file_path, format="PNG")
                    
                    # 添加图片路径到结果
                    result['image_paths'].append(temp_file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API处理失败: {str(e)}")
            raise e

    async def terminate(self):
        '''插件被卸载/停用时调用'''
        self.waiting_users.clear()
        
        # 清理临时文件
        if os.path.exists(self.temp_dir):
            try:
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
            except Exception as e:
                logger.error(f"清理临时文件时出错: {str(e)}")
