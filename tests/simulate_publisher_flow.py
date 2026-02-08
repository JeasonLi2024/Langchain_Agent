
import os
import sys
import asyncio
import base64
import json
import shutil
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_root = os.path.dirname(current_dir)
sys.path.insert(0, langchain_root)

# Load env and setup Django
from dotenv import load_dotenv
load_dotenv(os.path.join(langchain_root, ".env"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Project_Zhihui.settings")
from core.django_setup import setup_django
setup_django()

print("Importing modules...", flush=True)
# Import graph
from graph.publisher_main_agent import publisher_main_app
from project.models import Requirement, File
from django.conf import settings
print("Modules imported.", flush=True)

# --- Mocking for Test Robustness ---
from unittest.mock import MagicMock, patch

# MOCKING MILVUS FOR IMAGE TEST STABILITY
# Since we are focusing on image upload logic and Milvus might be flaky
try:
    from tools import search_tools
    # Mock retrieve_tags (Runnable)
    search_tools.retrieve_tags = MagicMock()
    async def async_return(*args, **kwargs):
        return "Mocked tags result"
    search_tools.retrieve_tags.ainvoke = MagicMock(side_effect=async_return)
    print("Mocked search_tools.retrieve_tags", flush=True)
except Exception as e:
    print(f"Failed to mock search_tools: {e}", flush=True)

from langgraph.checkpoint.memory import MemorySaver

# Inject MemorySaver for state persistence
checkpointer = MemorySaver()
publisher_main_app.checkpointer = checkpointer

# Mock for AI Image Generation
def mock_generate_poster_images(title, brief, tags, style='default', request=None):
    print(f"[MOCK] Generating images for title='{title}', style='{style}'")
    # Create dummy images in cover/tmp
    tmp_dir = os.path.join(settings.MEDIA_ROOT, "cover", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    urls = []
    batch_id = "mockbatch"
    for i in range(4):
        filename = f"{batch_id}_{i}.png"
        filepath = os.path.join(tmp_dir, filename)
        # Create a dummy colored image or just text file renamed to png
        with open(filepath, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100) # Dummy PNG header
        
        # Construct URL (Assuming MEDIA_URL is /media/)
        # We need to match what build_media_url does, usually /media/cover/tmp/...
        url = f"{settings.MEDIA_URL}cover/tmp/{filename}"
        urls.append(url)
    
    return urls

async def simulate_publisher_flow():
    print("=== Starting Publisher Flow Simulation ===", flush=True)
    
    # 1. Prepare File Input
    pdf_path = "/mnt/data/langchain-v2.0/tests/1. 华为云计算技术有限公司——“基于端云算力协同的疲劳驾驶智能识别”比赛方案_1-6.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}", flush=True)
        return

    print(f"Reading file: {pdf_path}", flush=True)
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    pdf_b64 = base64.b64encode(pdf_data).decode('utf-8')
    
    # Message 1: Upload File + Intent
    msg1_content = [
        {"type": "text", "text": "这是比赛方案文档，请帮我发布一个新的需求。"},
        {
            "type": "file", 
            "data": pdf_b64, 
            "mimeType": "application/pdf", 
            "name": os.path.basename(pdf_path)
        }
    ]
    
    # Initialize State
    config = {"configurable": {"thread_id": "test_sim_cover_001"}}
    
    print("\n--- Step 1: Sending File & Intent ---")
    inputs = {
        "messages": [HumanMessage(content=msg1_content)],
        "user_info": {"id": 556, "org_id": 6} 
    }
    
    async for event in publisher_main_app.astream(inputs, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            if "messages" in node_state:
                last_msg = node_state["messages"][-1]
                print(f"Output: {last_msg.content[:200]}..." if len(str(last_msg.content)) > 200 else f"Output: {last_msg.content}")

    print("\n--- Step 2: Confirm Draft & Ask for Tags ---")
    inputs_2 = {
        "messages": [HumanMessage(content="信息确认无误。")]
    }
    
    async for event in publisher_main_app.astream(inputs_2, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            if "messages" in node_state:
                for m in node_state["messages"]:
                    print(f"Msg ({m.type}): {m.content[:100]}...")

    print("\n--- Step 3: Request Tag Recommendation ---")
    inputs_3 = {
        "messages": [HumanMessage(content="需要推荐标签")]
    }
    
    async for event in publisher_main_app.astream(inputs_3, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            if "messages" in node_state:
                 for m in node_state["messages"]:
                    print(f"Msg ({m.type}): {m.content[:200]}...")
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        print(f"TOOL CALL DETECTED: {m.tool_calls}")

    print("\n--- Step 3.5: Request AI Cover Generation ---")
    # Real AI call - No patching
    # with patch('tools.ai_utils.generate_poster_images', side_effect=mock_generate_poster_images):
    if True:
        inputs_cover_ai = {
            "messages": [HumanMessage(content="帮我生成一张科技风格的封面")]
        }
        
        async for event in publisher_main_app.astream(inputs_cover_ai, config=config):
            for node_name, node_state in event.items():
                print(f"\n[Node: {node_name}]")
                if "messages" in node_state:
                     for m in node_state["messages"]:
                        print(f"Msg ({m.type}): {m.content[:200]}...")
                        if hasattr(m, 'tool_calls') and m.tool_calls:
                            print(f"TOOL CALL DETECTED: {m.tool_calls}")

    print("\n--- Step 3.5b: Select AI Cover ---")
    # User selects the first one
    # We need to capture the URL from the previous tool output in real execution
    # For simulation simplicity, we just say "select the first one" and let the agent handle it.
    # But wait, in the simulation loop, we don't automatically capture the URL.
    # However, since we are observing the output, we can't easily extract it programmatically 
    # without adding logic inside the loop.
    # BUT, the Agent is stateful. If we say "select the first one", the agent logic *should* know which one it is
    # IF the agent stored it in state. 
    # Looking at cover_flow_node: new_state['cover_image_candidates'] = image_urls
    # So the agent DOES know.
    # But `select_cover_image` tool takes a URL.
    # The Agent usually calls this tool with the URL it thinks is "first".
    
    # Let's try to just give the instruction and see if the Agent calls the tool correctly.
    # We don't need to pass the URL in our HumanMessage.
    
    inputs_select_ai = {
        "messages": [HumanMessage(content=f"我选第一张")] 
    }
    
    async for event in publisher_main_app.astream(inputs_select_ai, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            if "messages" in node_state:
                 for m in node_state["messages"]:
                    print(f"Msg ({m.type}): {m.content[:200]}...")
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        print(f"TOOL CALL DETECTED: {m.tool_calls}")

    print("\n--- Step 3.6: User Uploads Local Cover ---")
    # Prepare local image
    upload_img_path = "/mnt/data/langchain-v2.0/tests/Screenshot 2026_1_24 15_35_10.png"
    if not os.path.exists(upload_img_path):
        print(f"Warning: Test image not found at {upload_img_path}. Creating dummy.", flush=True)
        with open(upload_img_path, "wb") as f:
             f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100)
    
    with open(upload_img_path, "rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    msg_upload = [
        # {"type": "text", "text": "还是用我自己这张图吧"},
        {
            "type": "file", 
            "data": img_b64, 
            "mimeType": "image/png", 
            "name": os.path.basename(upload_img_path)
        }
    ]
    
    inputs_upload = {
        "messages": [HumanMessage(content=msg_upload)]
    }
    
    async for event in publisher_main_app.astream(inputs_upload, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            # Verify router injected system message
            if "messages" in node_state:
                 for m in node_state["messages"]:
                    print(f"Msg ({m.type}): {m.content[:200]}...")
                    
    print("\n--- Step 3.6b: Confirm Uploaded Cover ---")
    # Agent might ask "Do you want to use this image?". User says "Yes".
    # Or user just says "Use this".
    # We assume user intent is clear.
    inputs_confirm_upload = {
        "messages": [HumanMessage(content="确认使用这张上传的图片作为封面")]
    }
    
    async for event in publisher_main_app.astream(inputs_confirm_upload, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            if "messages" in node_state:
                 for m in node_state["messages"]:
                    print(f"Msg ({m.type}): {m.content[:200]}...")
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        print(f"TOOL CALL DETECTED: {m.tool_calls}")

    print("\n--- Step 4: Publish ---")
    inputs_4 = {
        "messages": [HumanMessage(content="确认发布")]
    }
    
    final_req_id = None
    
    async for event in publisher_main_app.astream(inputs_4, config=config):
        for node_name, node_state in event.items():
            print(f"\n[Node: {node_name}]")
            if "final_requirement_id" in node_state and node_state["final_requirement_id"]:
                final_req_id = node_state["final_requirement_id"]
                print(f"*** PUBLISH SUCCESS! ID: {final_req_id} ***")

    from asgiref.sync import sync_to_async

    # Verification
    if final_req_id:
        print("\n--- Verification ---")
        try:
            @sync_to_async
            def verify_req(req_id):
                req = Requirement.objects.get(id=req_id)
                status = req.status
                title = req.title
                cover = req.cover
                files_info = []
                for f in req.files.all():
                    files_info.append((f.name, f.real_path))
                return status, title, cover, files_info

            status, title, cover, files_info = await verify_req(final_req_id)
            print(f"DB Requirement Status: {status}")
            print(f"DB Title: {title}")
            print(f"DB Cover Field: {cover}")
            
            # Check Cover File Existence
            if cover:
                cover_abs_path = os.path.join(settings.MEDIA_ROOT, str(cover))
                print(f"Cover File Path: {cover_abs_path}")
                print(f"Cover File Exists: {os.path.exists(cover_abs_path)}")
                
                # Check cleanup: mockbatch_*.png should be gone from cover/tmp (except the selected one if we selected AI, but we switched to upload)
                # Since we switched to upload, the AI images in cover/tmp might still be there OR cleaned up if we selected one first.
                # Logic: We selected AI first -> Moved mockbatch_0.png to cover/YYYY/MM -> Cleaned others.
                # Then we uploaded -> Moved upload.png to cover/YYYY/MM -> Updated req.cover.
                # So mockbatch_0.png (renamed) should exist in cover/YYYY/MM (orphaned but existing), others gone.
                # And Uploaded image exists in cover/YYYY/MM.
                pass
            
            # Check Files
            print(f"Linked Files: {len(files_info)}")
            for name, real_path in files_info:
                print(f"- File: {name}, Path: {real_path}, Exists: {os.path.exists(os.path.join(settings.MEDIA_ROOT, real_path))}")
                
            print("\n--- Milvus Verification ---")
            from pymilvus import connections, Collection, utility
            from project.services import COLLECTION_EMBEDDINGS, COLLECTION_RAW_DOCS
            
            connections.connect(alias="default", host="localhost", port="19530")
            
            for name in [COLLECTION_EMBEDDINGS, COLLECTION_RAW_DOCS]:
                if utility.has_collection(name):
                    col = Collection(name)
                    col.load()
                    res_ids = col.query(expr=f"project_id == {final_req_id}", output_fields=["id", "content"])
                    print(f"Collection {name}: Found {len(res_ids)} entries for Req {final_req_id}")
                    if len(res_ids) > 0:
                        print(f"Sample content: {res_ids[0].get('content')[:100]}...")
                else:
                    print(f"Collection {name} NOT FOUND")

        except Exception as e:
            print(f"Verification Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simulate_publisher_flow())
