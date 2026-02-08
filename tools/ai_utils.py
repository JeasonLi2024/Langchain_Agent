import os
import logging
import requests
import uuid
from django.conf import settings
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions, OptimizePromptOptions
# from common_utils import build_media_url

logger = logging.getLogger(__name__)

# 风格提示词映射
STYLE_PROMPTS = {
    'default': '现代商务或科技极简风格，画面清晰，构图平衡，适合项目展示',
    'tech': '赛博朋克风格，霓虹灯光，未来城市，高科技感，深色背景，酷炫',
    'illustration': '扁平化矢量插画，色彩明亮，简约线条，创意几何图形，适合UI设计',
    'ink': '中国水墨画风格，写意，留白，山水意境，传统美学，大气',
    '3d': 'C4D渲染风格，3D立体模型，柔和光照，材质细腻，现代感，抽象艺术'
}

import concurrent.futures

# ... existing code ...

def download_image(url, temp_dir, idx, batch_id, request=None):
    """
    下载单张图片的辅助函数
    """
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            file_name = f"{batch_id}_{idx}.png"
            file_path = os.path.join(temp_dir, file_name)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            relative_path = f"cover/tmp/{file_name}"
            
            # Enforce absolute URL with specific domain as requested
            base_domain = "https://zhihui.bupt.edu.cn"
            media_prefix = settings.MEDIA_URL
            if not media_prefix.startswith("/"):
                media_prefix = "/" + media_prefix
            if not media_prefix.endswith("/"):
                media_prefix = media_prefix + "/"
                
            return f"{base_domain}{media_prefix}{relative_path}"
        else:
            logger.error(f"Failed to download image from {url}, status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception downloading image from {url}: {e}")
        return None

def generate_poster_images(title, brief, tags, style='default', request=None):
    """
    调用 Doubao-Seedream-4.5 生成海报图片 (使用 volcenginesdkarkruntime)
    并实时并行下载图片
    
    Args:
        title (str): 需求标题
        brief (str): 需求简介
        tags (str/list): 标签
        style (str): 风格代码
        request: 请求对象，用于构建本地URL
        
    Returns:
        list: 本地图片URL列表
    """
    # 从环境变量获取 API Key
    api_key = os.environ.get('ARK_API_KEY')
    
    # 兼容性保留：如果没有环境变量，尝试从 settings 获取
    if not api_key:
        api_key = getattr(settings, 'ARK_API_KEY', None)
        
    if not api_key:
        logger.error("未配置 ARK_API_KEY")
        raise Exception("未配置 AI 服务密钥 (ARK_API_KEY)")

    # 初始化客户端
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key
    )
    
    # 构建 Prompt
    style_desc = STYLE_PROMPTS.get(style, STYLE_PROMPTS['default'])
    tags_str = tags if isinstance(tags, str) else ' '.join(tags)
    
    prompt = (
        f"设计一组（4张）专业的项目需求横板海报封面，提供不同的设计方案供选择。\n"
        f"【项目标题】：{title}\n"
        f"【项目简介】：{brief}\n"
        f"【相关标签】：{tags_str}\n"
        f"【画面要求】：\n"
        f"1. 风格：{style_desc}。\n"
        f"2. 内容：抽象化表达项目主题，避免过多的文字，突出该项目需求的核心概念。\n"
        f"3. 构图：中心构图或左右构图，留有适当留白，视觉冲击力强。\n"
        f"4. 质量：高分辨率，细节丰富，无模糊噪点。"
    )
    
    # 准备下载目录
    temp_rel_dir = os.path.join('cover', 'tmp')
    temp_dir = os.path.join(settings.MEDIA_ROOT, temp_rel_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    local_urls = []
    futures = []
    
    # 生成统一的 batch_id，用于后续清理同批次未选中的图片
    batch_id = uuid.uuid4().hex
    
    try:
        # 调用图片生成 API
        model_id = os.environ.get('ARK_MODEL_ENDPOINT', "doubao-seedream-4-0-250828")
        logger.info(f"Generating poster with model: {model_id}")
        
        stream = client.images.generate(
            model=model_id,
            prompt=prompt,
            size="2K",
            response_format="url",
            watermark=False,
            sequential_image_generation="auto",
            sequential_image_generation_options=SequentialImageGenerationOptions(max_images=4),
            stream=True,
            optimize_prompt_options=OptimizePromptOptions(mode="fast")
        )
        
        image_idx = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for event in stream:
                if event is None:
                    continue
                if event.type == "image_generation.partial_failed":
                    logger.error(f"Stream generate images error: {event.error}")
                    if event.error is not None:
                         code = event.error.code
                         if hasattr(code, 'equal'):
                             if code.equal("InternalServiceError"):
                                 break
                         elif str(code) == "InternalServiceError":
                             break
                elif event.type == "image_generation.partial_succeeded":
                    if event.error is None and event.url:
                        logger.info(f"recv.Size: {event.size}, recv.Url: {event.url}")
                        # 提交下载任务，传入 batch_id
                        futures.append(executor.submit(download_image, event.url, temp_dir, image_idx, batch_id, request))
                        image_idx += 1
                elif event.type == "image_generation.completed":
                    if event.error is None:
                        logger.info(f"Final completed event. Usage: {event.usage}")
            
            # 等待所有下载任务完成
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    local_urls.append(result)
        
        if not local_urls:
             logger.warning(f"API返回成功但无图片被成功下载")
             
        return local_urls

    except Exception as e:
        logger.error(f"海报生成服务异常: {e}")
        raise Exception(f"海报生成服务异常: {str(e)}")

# Remove the old save_temp_images function as it is now integrated or handle compatibility if needed
# Keeping a simplified version or alias if other parts use it, but logic is moved inside generate_poster_images
def save_temp_images(image_urls, request=None):
    # This function is kept for backward compatibility if needed, 
    # but the new flow handles downloading inside generate_poster_images
    return [] 

