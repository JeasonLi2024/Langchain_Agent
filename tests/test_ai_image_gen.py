
import os
import sys
import django
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_root = os.path.dirname(current_dir)
sys.path.insert(0, langchain_root)

# Load env
from dotenv import load_dotenv
load_dotenv(os.path.join(langchain_root, ".env"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Project_Zhihui.settings")
from core.django_setup import setup_django
setup_django()

from tools.ai_utils import generate_poster_images

logging.basicConfig(level=logging.INFO)

def test_generate_images():
    print("Testing generate_poster_images...")
    try:
        title = "星际穿越复古列车"
        brief = "黑洞里冲出一辆快支离破碎的复古列车"
        tags = ["科幻", "电影感", "超现实"]
        style = "tech"
        
        urls = generate_poster_images(title, brief, tags, style=style)
        
        print("\n=== Generation Result ===")
        if urls:
            print(f"Successfully generated {len(urls)} images:")
            for url in urls:
                print(f"- {url}")
        else:
            print("No images generated (list is empty).")
            
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate_images()
