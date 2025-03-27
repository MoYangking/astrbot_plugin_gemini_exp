import base64
import mimetypes # Added import
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.message_components import *
import asyncio
import sys
import importlib
import requests
from io import BytesIO
import time
import tempfile
import os
# Make sure google.genai is correctly imported
try:
    from google import genai
    from google.genai import types
    from google.genai.types import HarmCategory, HarmBlockThreshold # Import specific enums if needed
    from google.genai.types import HttpOptions
except ImportError:
    # Handle the case where the library might not be installed yet during check/install
    genai = None
    types = None
    HarmCategory = None
    HarmBlockThreshold = None
    HttpOptions = None
    print("google.generativeai not found initially, attempting installation.")


from PIL import Image as PILImage
from astrbot.core.utils.io import download_image_by_url


@register("astrbot_plugin_geminiexp", "YourName", "基于 Google Gemini 2.0 Flash Experimental 多模态模型的插件", "1.0.0")
class GeminiExpPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.api_key = config.get("api_key", "")
        self.http_proxy = config.get("http_proxy", None) # Optional proxy support
        self.base_url = config.get("base_url", None) # Optional custom base URL
        self.waiting_users = {}  # 存储正在等待输入的用户 {user_id: expiry_time}
        self.temp_dir = tempfile.mkdtemp(prefix="gemini_exp_")
        self.client = None # Initialize client later

        # Check and install necessary packages
        if not self._check_packages():
            self._install_packages()
            # Reload modules after installation
            self._reload_modules()

        # Configure Gemini client after potential installation
        self._configure_gemini()

    def _check_packages(self) -> bool:
        """检查是否安装了需要的包"""
        try:
            importlib.import_module('google.generativeai') # More specific check
            importlib.import_module('PIL')
            return True
        except ImportError:
            return False

    def _install_packages(self):
        """安装必要的包"""
        try:
            import subprocess
            logger.info("开始安装 google-generativeai 和 pillow...")
            # Ensure pip uses the correct python environment
            pip_command = [sys.executable, "-m", "pip", "install", "-U", "google-generativeai", "pillow"]
            if self.http_proxy:
                 pip_command.extend(["--proxy", self.http_proxy])
            subprocess.check_call(pip_command)
            logger.info("成功安装必要的包")
        except subprocess.CalledProcessError as e:
            logger.error(f"安装包失败: {str(e)}")
            raise RuntimeError(f"无法安装必要的依赖项: {e}") # Re-raise as runtime error

    def _reload_modules(self):
        """重新加载安装后的模块"""
        global genai, types, HarmCategory, HarmBlockThreshold, HttpOptions, PILImage
        try:
            # Invalidate caches to ensure fresh import
            importlib.invalidate_caches()
            # Use importlib.reload if modules were already imported (though unlikely here)
            # Or just re-import
            google_module = importlib.import_module('google.generativeai')
            genai = google_module
            types = importlib.import_module('google.genai.types')
            HarmCategory = types.HarmCategory
            HarmBlockThreshold = types.HarmBlockThreshold
            HttpOptions = types.HttpOptions # Re-assign HttpOptions

            pil_module = importlib.import_module('PIL.Image')
            PILImage = pil_module
            logger.info("模块重新加载成功")
        except Exception as e:
            logger.error(f"重新加载模块失败: {e}")
            raise RuntimeError(f"无法重新加载依赖项: {e}")

    def _configure_gemini(self):
        """配置 Gemini 客户端"""
        if not self.api_key:
            logger.warning("Gemini API 密钥未配置!")
            return

        if not genai:
             logger.error("google.generativeai 模块未能加载。")
             return

        try:
            http_options = None
            if self.base_url:
                 # Note: As of recent versions, base_url might be part of Client constructor
                 # or transport options, check library docs if HttpOptions doesn't work.
                 # This HttpOptions might be deprecated or work differently now.
                 # Let's try configuring transport directly if available, else fallback.
                 logger.info(f"使用自定义 Base URL: {self.base_url}")
                 # http_options = HttpOptions(base_url=self.base_url) # This might be old way

            # Configure transport for proxy and potentially base_url
            transport = None
            client_options=None
            if self.http_proxy or self.base_url:
                # Check if client_options is supported for newer library versions
                client_options = {}
                if self.http_proxy:
                    client_options['api_endpoint'] = self.base_url # Newer way? Check docs
                    # Proxy might need to be configured via environment variables (HTTPS_PROXY)
                    # or potentially a custom transport object. Setting env var is often easier.
                    os.environ['HTTPS_PROXY'] = self.http_proxy
                    os.environ['HTTP_PROXY'] = self.http_proxy
                    logger.info(f"设置代理环境变量: {self.http_proxy}")
                if self.base_url:
                    # For custom endpoints like LMStudio or Ollama proxies via litellm:
                    client_options['api_endpoint'] = self.base_url
                    logger.info(f"设置 API 端点: {self.base_url}")


            # Initialize the client
            self.client = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp-image-generation", # <-- Use the specific model
                api_key=self.api_key,
                # Pass client_options if using newer style, else None
                client_options=client_options if client_options else None
                # transport=transport # Pass custom transport if needed/possible
            )
            logger.info("Gemini 客户端配置成功，使用模型: gemini-2.0-flash-exp-image-generation")

        except Exception as e:
            logger.error(f"配置 Gemini 客户端失败: {e}")
            self.client = None


    @filter.command("gemexp")
    async def gemini_exp(self, event: AstrMessageEvent):
        '''使用Gemini 2.0 Flash Experimental模型进行多模态交互'''
        if not self.api_key:
            yield event.plain_result("请联系管理员配置Gemini API密钥")
            return
        if not self.client:
             yield event.plain_result("Gemini 客户端未初始化，请检查配置和日志。")
             return

        user_id = event.get_sender_id()
        user_name = event.get_sender_name()

        self.waiting_users[user_id] = time.time() + 60
        yield event.plain_result(f"好的 {user_name}，请在60秒内发送您的文本描述和图片（如有），我将尝试使用 Gemini 修改或生成图片。")


    @filter.event_message_type(EventMessageType.ALL)
    async def handle_follow_up(self, event: AstrMessageEvent):
        """处理用户的后续消息 (包括文本和图片)"""
        if not isinstance(event, AstrMessageEvent):
            logger.error(f"handle_follow_up 收到了错误类型的参数: {type(event)}")
            return

        user_id = event.get_sender_id()
        message_text = event.message_str.strip()

        # 忽略所有命令消息 (简单检查)
        if message_text.startswith("/"):
            # Only ignore if it's *not* the user we are waiting for,
            # or if it *is* the user but the command isn't related
            # (This check might be too simple)
            if user_id in self.waiting_users:
                 # Allow user to potentially send commands while waiting? Or cancel?
                 # For now, let's just process if they are the waiting user.
                 pass
            else:
                 return

        if user_id not in self.waiting_users:
            return

        if time.time() > self.waiting_users[user_id]:
            del self.waiting_users[user_id]
            # Don't send timeout message if they actually sent something, just process it
            # yield event.plain_result("等待超时，请重新发送命令。")
            # return # Let it proceed if they sent data just after timeout? Or enforce? Let's enforce.
            logger.info(f"用户 {user_id} 输入超时。")
            # yield event.plain_result("等待超时，请重新发送命令。") # Maybe annoying?
            return


        # Process the message, remove user from waiting list
        del self.waiting_users[user_id]
        logger.info(f"处理来自用户 {user_id} 的后续消息。")

        message_chain = event.get_messages()
        text_content = event.message_str # Get combined text
        image_list_pil = [] # Store PIL images

        # Extract images from message chain
        for msg in message_chain:
            if isinstance(msg, Image):
                try:
                    img_url = None
                    if hasattr(msg, 'url') and msg.url:
                        img_url = msg.url

                    if img_url:
                        temp_img_path = await download_image_by_url(img_url)
                        if temp_img_path:
                            img = PILImage.open(temp_img_path)
                            # It's good practice to copy the image data
                            # as the underlying file might be temporary
                            image_list_pil.append(img.copy())
                            img.close() # Close the file handle
                            logger.info(f"成功处理来自URL的图片: {img_url}")
                            # Clean up downloaded temp file immediately if possible
                            # try:
                            #    os.remove(temp_img_path)
                            # except OSError as e:
                            #    logger.warning(f"无法删除临时下载文件 {temp_img_path}: {e}")
                        else:
                            logger.warning(f"下载图片失败: {img_url}")

                except Exception as e:
                    logger.error(f"处理消息链中的图片时出错: {str(e)}")
                    yield event.plain_result(f"无法处理收到的图片，请稍后再试。错误: {str(e)}")
                    return

        # Check if client is ready
        if not self.client:
             yield event.plain_result("Gemini 客户端未就绪，无法处理请求。请检查配置。")
             return

        # Ensure there's some content
        if not text_content and not image_list_pil:
            yield event.plain_result("请提供文本描述或图片。")
            return

        yield event.plain_result("正在处理您的请求，请稍候...")

        # Call Gemini API using the new structure
        try:
            # --- Modification Start ---
            # Prepare contents in the new format
            user_parts = []
            if text_content:
                 # Use types.Part.from_text as in the example
                 user_parts.append(types.Part.from_text(text=text_content))

            for img_pil in image_list_pil:
                # Convert PIL image to bytes (e.g., PNG)
                img_byte_arr = BytesIO()
                img_pil.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                # Add as inline data part
                user_parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)))

            if not user_parts:
                 raise ValueError("无法构建有效的 API 请求内容。")

            # Construct the final contents list
            gemini_contents = [types.Content(role="user", parts=user_parts)]

            # Define the generation config matching the second script
            # Using the second set of safety settings provided
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["image", "text"], # Match example
                safety_settings=[
                    # Match the second example's safety settings
                    types.SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, # Block most
                    ),
                    types.SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, # Block most
                    ),
                    # You might want to add other categories if needed, e.g., HATE_SPEECH, HARASSMENT
                     types.SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ],
                response_mime_type="text/plain", # Match example
                # Add other parameters like temperature if desired
                # temperature=0.7,
            )

            # Use generate_content (non-streaming) for simplicity in this plugin context
            # If streaming is essential, the response handling needs significant change
            # response = await self.client.generate_content_async(  <-- Use async version if available/needed
            # Let's stick to sync call within to_thread for now as client setup is sync
            response = await asyncio.to_thread(
                self.client.generate_content,
                contents=gemini_contents,
                generation_config=generate_content_config,
                # stream=False # Ensure non-streaming if default changed
            )

            # --- Modification End ---


            # Process the response (similar to original, but adapted for potential errors/blocks)
            result = {'text': '', 'image_paths': []}

            # Add more robust response checking
            if not response:
                raise ValueError("Gemini API 返回了空响应")

            # Check for safety blocks or empty candidates
            if not response.candidates:
                 prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
                 block_reason = "未知原因"
                 if prompt_feedback and hasattr(prompt_feedback, 'block_reason'):
                     block_reason = str(prompt_feedback.block_reason)
                 logger.warning(f"Gemini API 返回无有效候选内容，可能已被阻止。原因: {block_reason}")
                 safety_ratings_str = "\n".join([f"- {s.category}: {s.probability}" for s in prompt_feedback.safety_ratings]) if prompt_feedback and hasattr(prompt_feedback, 'safety_ratings') else "无安全评分信息"
                 yield event.plain_result(f"请求可能因安全策略被阻止 ({block_reason}) 或未生成有效内容。\n{safety_ratings_str}")
                 return


            # Assume first candidate is the primary one
            candidate = response.candidates[0]

            if not candidate.content or not candidate.content.parts:
                finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else '未知'
                safety_ratings_str = "\n".join([f"- {s.category}: {s.probability}" for s in candidate.safety_ratings]) if hasattr(candidate, 'safety_ratings') else "无安全评分信息"
                logger.warning(f"Gemini API 返回的候选内容为空。完成原因: {finish_reason}")
                yield event.plain_result(f"模型未能在响应中生成内容。完成原因: {finish_reason}\n安全评分:\n{safety_ratings_str}")
                return

            # Process parts (text and image)
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text is not None:
                    result['text'] += part.text
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    try:
                        img_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type
                        img = PILImage.open(BytesIO(img_data))

                        # Guess extension based on mime type
                        file_extension = mimetypes.guess_extension(mime_type) or ".png" # Default to png

                        # Create unique temp file path
                        timestamp = int(time.time() * 1000)
                        temp_file_path = os.path.join(self.temp_dir, f"gemini_result_{timestamp}{file_extension}")

                        img.save(temp_file_path) # Save in the format determined by PIL based on extension or default
                        result['image_paths'].append(temp_file_path)
                        logger.info(f"已保存 Gemini 生成的图片到: {temp_file_path}")
                    except Exception as img_err:
                         logger.error(f"处理 Gemini 返回的图片数据时出错: {img_err}")
                         result['text'] += f"\n(错误：无法处理返回的图片数据: {img_err})"


            # --- Response Sending Logic (mostly unchanged) ---
            text_response = result.get('text', '').strip()
            image_paths = result.get('image_paths', [])

            if not text_response and not image_paths:
                 yield event.plain_result("Gemini 模型没有返回任何文本或图片内容。")
                 return

            # Use Nodes for multiple images, otherwise simple chain
            if len(image_paths) >= 2:
                bot_id = self.config.get("bot_id", event.bot_id or 114514) # Use bot's actual ID if available
                bot_name = self.config.get("bot_name", "Gemini助手")

                # Simple split: Put all text before the first image, or just with the first image?
                # Let's put text with the first node, subsequent nodes are just images.
                # Or try to associate text chunks roughly - Keep original logic for now.

                # (Reusing your previous text splitting logic)
                text_parts = []
                if text_response and len(image_paths) > 0:
                    paragraphs = text_response.split('\n\n')
                    num_images = len(image_paths)
                    if len(paragraphs) >= num_images:
                        parts_per_section = len(paragraphs) // num_images
                        current_start = 0
                        for i in range(num_images):
                            if i < num_images - 1:
                                current_end = current_start + parts_per_section
                                text_parts.append('\n\n'.join(paragraphs[current_start:current_end]))
                                current_start = current_end
                            else: # Last image gets the rest
                                text_parts.append('\n\n'.join(paragraphs[current_start:]))
                    else: # Less paragraphs than images, fallback to character split
                        total_chars = len(text_response)
                        chars_per_image = total_chars // num_images
                        current_start = 0
                        for i in range(num_images):
                             if i < num_images - 1:
                                current_end = current_start + chars_per_image
                                # Try to find a better split point near the target end
                                best_split = current_end
                                for j in range(current_end, min(current_end + 50, total_chars)):
                                    if text_response[j] in ['.', '?', '!', '。', '？', '！', '\n']:
                                        best_split = j + 1
                                        break
                                text_parts.append(text_response[current_start:best_split])
                                current_start = best_split
                             else: # Last image gets the rest
                                text_parts.append(text_response[current_start:])
                elif text_response: # Text but no images? Should not happen if modality includes image
                     text_parts = [text_response] + [''] * (len(image_paths) -1) # Put all text first
                else: # Images but no text
                     text_parts = [''] * len(image_paths)


                # Ensure text_parts matches image_paths length
                if len(text_parts) < len(image_paths):
                    text_parts.extend([''] * (len(image_paths) - len(text_parts)))
                elif len(text_parts) > len(image_paths):
                     text_parts = text_parts[:len(image_paths)-1] + ['\n\n'.join(text_parts[len(image_paths)-1:])]


                # Create Nodes
                ns = Nodes([])
                for i, img_path in enumerate(image_paths):
                    node_text = text_parts[i].strip()
                    # Add image number prefix maybe?
                    # prefix = f"图 {i+1}/{len(image_paths)}:\n" if len(image_paths) > 1 else ""
                    prefix = "" # Keep it simple for now

                    node_content = []
                    if node_text:
                        node_content.append(Plain(prefix + node_text))
                    node_content.append(Image.fromFileSystem(img_path))

                    ns.nodes.append(
                        Node(uin=bot_id, name=bot_name, content=node_content)
                    )

                yield event.chain_result([ns]) # Send as Forward Message

            else: # 0 or 1 image
                chain = []
                if text_response:
                    chain.append(Plain(text_response))
                for img_path in image_paths:
                    chain.append(Image.fromFileSystem(img_path))

                if chain:
                     yield event.chain_result(chain)
                # else: # Should have been caught earlier
                #    yield event.plain_result("Gemini did not return usable content.")


        except Exception as e:
            logger.error(f"Gemini API 调用或处理失败: {str(e)}", exc_info=True) # Log traceback
            yield event.plain_result(f"处理失败: {str(e)}")


    async def terminate(self):
        '''插件被卸载/停用时调用'''
        self.waiting_users.clear()

        # Clean up temporary files
        if os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"已清理临时目录: {self.temp_dir}")
            except Exception as e:
                logger.error(f"清理临时文件时出错: {str(e)}")
