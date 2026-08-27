import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv


STYLE_PROMPTS: Dict[str, str] = {
    "default": "现代商务或科技极简风格，画面清晰，构图平衡，适合项目展示",
    "tech": "赛博朋克风格，霓虹灯光，未来城市，高科技感，深色背景，酷炫",
    "illustration": "扁平化矢量插画，色彩明亮，简约线条，创意几何图形，适合 UI 设计",
    "ink": "中国水墨画风格，写意，留白，山水意境，传统美学，大气",
    "3d": "C4D 渲染风格，3D 立体模型，柔和光照，材质细腻，现代感，抽象艺术",
}

DEFAULT_TITLE = "基于多模态智能体的校园项目协作平台"
DEFAULT_BRIEF = (
    "面向高校创新实践场景，构建一个支持项目发布、智能推荐、海报生成与资料检索的"
    "多模态项目协作平台。"
)
DEFAULT_TAGS = ["AI", "智能体", "项目推荐", "校园协作"]
DEFAULT_SIZE = "1024x768"
DEFAULT_COUNT = 4
AGNES_ENDPOINT = "https://apihub.agnes-ai.com/v1/images/generations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Agnes Image 2.1 Flash 测试与 Doubao Seedream 相同的项目海报生图任务。"
    )
    parser.add_argument("--title", default=DEFAULT_TITLE, help="项目标题")
    parser.add_argument("--brief", default=DEFAULT_BRIEF, help="项目简介")
    parser.add_argument(
        "--tags",
        default=",".join(DEFAULT_TAGS),
        help="标签列表，使用逗号分隔，例如 AI,Agent,Poster",
    )
    parser.add_argument(
        "--style",
        default="default",
        choices=sorted(STYLE_PROMPTS.keys()),
        help="风格代码",
    )
    parser.add_argument("--size", default=DEFAULT_SIZE, help="Agnes 输出尺寸，例如 1024x768")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="候选海报数量。Agnes 接口按单张生成，这里通过多次请求模拟原 Seedream 的 4 张方案输出。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="单次请求超时时间（秒）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output" / "agnes_image"),
        help="结果保存目录",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载返回的图片 URL 到本地",
    )
    return parser.parse_args()


def load_environment() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def get_api_key() -> str:
    api_key = (os.getenv("Agnes_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("未在 .env 中找到 Agnes_API_KEY")
    return api_key


def get_model_name() -> str:
    return (os.getenv("Agnes_Image_MODEL") or "agnes-image-2.1-flash").strip()


def normalize_tags(raw_tags: str) -> List[str]:
    items = [item.strip() for item in re.split(r"[,，]", raw_tags) if item.strip()]
    return items or DEFAULT_TAGS


def build_prompt(title: str, brief: str, tags: List[str], style: str, variant_index: int) -> str:
    style_desc = STYLE_PROMPTS.get(style, STYLE_PROMPTS["default"])
    tags_str = " / ".join(tags)
    variant_hint = (
        f"这是同一项目海报方案中的第 {variant_index} 张，请在保持核心信息一致的前提下，"
        f"提供与其他方案明显不同的视觉设计方向。"
    )

    return (
        "Design a professional horizontal cover poster for a university project requirement showcase. "
        "The image should be suitable for a project publishing platform hero banner. "
        f"Project title: {title}. "
        f"Project brief: {brief}. "
        f"Related tags: {tags_str}. "
        f"Style direction: {style_desc}. "
        f"{variant_hint} "
        "Use abstract visual storytelling to express the project theme, avoid dense readable text, "
        "preserve strong visual hierarchy, keep enough whitespace for future title overlay, "
        "prefer center or left-right balanced composition, cinematic lighting, high detail, "
        "clean structure, sharp focus, no blur, no watermark, high information density, "
        "suitable for a modern tech or innovation platform."
    )


def build_payload(model: str, prompt: str, size: str) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "size": size,
        "extra_body": {
            "response_format": "url",
        },
    }


def request_image(api_key: str, payload: dict, timeout: int) -> dict:
    response = requests.post(
        AGNES_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_url(response_json: dict) -> str:
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"响应中缺少 data 列表: {response_json}")
    url = data[0].get("url")
    if not url:
        raise ValueError(f"响应中缺少图片 URL: {response_json}")
    return url


def maybe_download(url: str, output_path: Path, timeout: int) -> None:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def main() -> int:
    args = parse_args()
    load_environment()

    api_key = get_api_key()
    model = get_model_name()
    tags = normalize_tags(args.tags)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in range(1, args.count + 1):
        prompt = build_prompt(args.title, args.brief, tags, args.style, idx)
        payload = build_payload(model, prompt, args.size)

        started_at = time.time()
        response_json = request_image(api_key, payload, args.timeout)
        elapsed = round(time.time() - started_at, 2)
        image_url = extract_url(response_json)

        item = {
            "index": idx,
            "elapsed_seconds": elapsed,
            "model": model,
            "size": args.size,
            "style": args.style,
            "image_url": image_url,
            "prompt": prompt,
            "raw_response": response_json,
        }

        if args.download:
            image_path = output_dir / f"agnes_poster_{idx}.png"
            maybe_download(image_url, image_path, args.timeout)
            item["downloaded_file"] = str(image_path)

        results.append(item)
        print(f"[OK] variant={idx} elapsed={elapsed}s url={image_url}")

    result_file = output_dir / "agnes_generation_results.json"
    result_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到: {result_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
