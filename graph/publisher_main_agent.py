import json
import sys
import os
import django
import base64
import tempfile
import uuid

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Django via core.django_setup
from core.django_setup import setup_django
setup_django()

from typing import TypedDict, Annotated, List, Literal, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from core.config import Config
from core.prompts import PUBLISHER_ROUTER_PROMPT, PUBLISHER_CHAT_SYSTEM_PROMPT
from graph.publisher_agent import publisher_app
from graph.file_parsing_graph import file_parsing_app
from core.embedding_service import generate_embedding, get_or_create_collection, COLLECTION_EMBEDDINGS, COLLECTION_RAW_DOCS
from project.services import delete_requirement_vectors, sync_raw_docs_from_text
from project.models import Requirement

# Import custom PickleRedisSaver
from core.persistence import PickleRedisSaver
import redis

# --- 1. Define State ---
class PublisherMasterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: str
    user_info: dict
    
    # Publisher Context
    file_path: Optional[str] # If user uploaded a file
    original_filename: Optional[str] # If user uploaded a file, keep the original name
    parsed_file_data: Optional[dict] # Data from file_parsing_graph
    publisher_state: Optional[dict] # State returned from publisher_agent
    
    # Results
    final_requirement_id: Optional[int]
    final_requirement_data: Optional[Dict[str, Any]] # For vector sync

# --- 2. Define Nodes ---

def cleanup_stale_files():
    """
    Clean up stale files in the tmp directory that are older than 1 hour.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tmp_dir = os.path.join(project_root, "tmp")
        if not os.path.exists(tmp_dir):
            return
            
        import time
        now = time.time()
        expiration_time = 3600 # 1 hour
        
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            if os.path.isfile(file_path):
                # Check modification time
                if now - os.path.getmtime(file_path) > expiration_time:
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up stale file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting stale file {file_path}: {e}")
    except Exception as e:
        print(f"Error during stale file cleanup: {e}")

async def router_node(state: PublisherMasterState):
    """
    Router for Publisher Agent.
    Decides entry point based on file upload or user intent.
    Handles Multimodal Messages (Base64 file uploads).
    """
    # Trigger cleanup opportunistically
    # cleanup_stale_files() # Removed to avoid sync blocking, rely on background task
    
    file_path = state.get("file_path")
    original_filename = state.get("original_filename")
    messages = state["messages"]
    new_file_uploaded = False
    
    # Check for multimodal message with file data in the last message
    if messages and isinstance(messages[-1].content, list):
        for item in messages[-1].content:
            if isinstance(item, dict) and (item.get('type') == 'file' or item.get('type') == 'image_url'): # Check for image_url too as some UIs use it for files
                # Found a file upload!
                file_data_b64 = item.get('data') or item.get('content') # Support 'content' key too
                
                if file_data_b64:
                    try:
                            # Decode and save to a local 'tmp' directory for better persistence
                            # Ideally, this should be in the project root to ensure it's accessible and manageable
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            tmp_dir = os.path.join(project_root, "tmp")
                            os.makedirs(tmp_dir, exist_ok=True)
                            
                            # Detect extension
                            decoded_data = base64.b64decode(file_data_b64)
                            ext = ".pdf" # Default
                            
                            # Try to get extension from mimeType first
                            mime_type = item.get('mimeType', '')
                            if 'pdf' in mime_type:
                                ext = ".pdf"
                            elif 'word' in mime_type or 'doc' in mime_type:
                                ext = ".docx" # Generic docx
                            elif decoded_data.startswith(b'%PDF'):
                                ext = ".pdf"
                            elif decoded_data.startswith(b'\xd0\xcf\x11\xe0'): # DOC
                                ext = ".doc"
                            elif decoded_data.startswith(b'PK\x03\x04'): # DOCX/ZIP
                                ext = ".docx"
                                
                            temp_filename = f"upload_{uuid.uuid4()}{ext}"
                            temp_path = os.path.join(tmp_dir, temp_filename)
                            
                            with open(temp_path, "wb") as f:
                                f.write(decoded_data)
                                
                            # Update state
                            file_path = temp_path
                            # Try to get filename from various keys, fallback to uuid
                            # UPDATE: Based on user logs, 'metadata' key exists. Let's check inside it.
                            metadata = item.get('metadata', {})
                            
                            original_filename = (
                                item.get('name') or 
                                item.get('filename') or 
                                item.get('fileName') or 
                                item.get('file_name') or
                                item.get('title') or
                                item.get('label') or
                                item.get('path') or
                                # Check metadata
                                metadata.get('name') or
                                metadata.get('filename') or
                                metadata.get('fileName') or
                                metadata.get('title')
                            )
                            
                            # Heuristic: If still no valid name, search all values for a string ending in .pdf/.doc/.docx
                            if not original_filename:
                                for k, v in item.items():
                                    if k != 'data' and k != 'content' and isinstance(v, str) and (v.lower().endswith('.pdf') or v.lower().endswith('.doc') or v.lower().endswith('.docx')):
                                        original_filename = os.path.basename(v)
                                        break
                                        
                            # Extreme Fallback: Check if the 'text' field contains a filename-like string
                            if not original_filename:
                                 text_val = item.get('text', '')
                                 if text_val and (text_val.lower().endswith('.pdf') or text_val.lower().endswith('.doc') or text_val.lower().endswith('.docx')):
                                      original_filename = os.path.basename(text_val)

                            if not original_filename:
                                 original_filename = f"upload_{uuid.uuid4()}{ext}"
                            else:
                                 # Ensure it's just the basename if it was a path
                                 original_filename = os.path.basename(original_filename)

                            # Remove "upload_" prefix if it's not a fallback uuid (basic check)
                            if original_filename.startswith("upload_") and len(original_filename) > 40 and "uuid" not in original_filename:
                                 # It might be a valid name starting with upload_, keep it.
                                 pass
                                 
                            new_file_uploaded = True
                            print(f"File uploaded: {original_filename} saved to: {file_path}") # Log for debugging
                            
                    except Exception as e:
                        print(f"Error decoding file: {e}")

    last_msg = ""
    if messages:
        if isinstance(messages[-1].content, str):
            last_msg = messages[-1].content
        elif isinstance(messages[-1].content, list):
            # Extract text from multimodal
            text_parts = [item.get('text', '') for item in messages[-1].content if item.get('type') == 'text']
            last_msg = " ".join(text_parts)
    
    # Priority 1: File Upload -> Parsing
    # Only route to parsing if it's a NEW file or we have a file but no parsed data yet
    if new_file_uploaded or (file_path and not state.get("parsed_file_data")):
        return {"next_step": "file_parsing_flow", "file_path": file_path, "original_filename": original_filename}
    
    # Priority 2: Check if we are already in Publisher Flow
    # Optimization: If we have an active draft or publisher state is incomplete,
    # skip LLM classification and route directly to publisher_flow.
    publisher_state = state.get("publisher_state", {})
    if publisher_state and not publisher_state.get("is_complete", False):
        # We assume if there is a publisher state and it's not complete, we are in the flow.
        # However, we should also check if the user wants to exit?
        # But for "Turn 2" latency optimization, we assume continuation.
        return {"next_step": "publisher_flow"}
    
    # Priority 3: Publisher Flow (Explicit Intent or Continuing)
    llm = Config.get_utility_llm()
    prompt = PUBLISHER_ROUTER_PROMPT
    try:
        if len(messages) < 2 and len(last_msg) < 10: # Simple heuristic for short greetings
             # Let's use LLM to be safe
             pass
             
        response = llm.invoke(prompt.format(message=last_msg))
        intent = response.content.strip().upper()
        
        if "PUBLISH" in intent:
            return {"next_step": "publisher_flow"}
        else:
            return {"next_step": "chat_node"}
    except:
        return {"next_step": "chat_node"}

def chat_node(state: PublisherMasterState):
    """
    Simple Chat / Greeting Node.
    """
    llm = Config.get_utility_llm()
    messages = state["messages"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PUBLISHER_CHAT_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return {"messages": [response]}

def file_parsing_node(state: PublisherMasterState):
    """
    Handle file parsing.
    """
    file_path = state["file_path"]
    original_filename = state.get("original_filename") or os.path.basename(file_path)
    
    input_state = {
        "file_path": file_path,
        "file_name": original_filename,
    }
    
    result = file_parsing_app.invoke(input_state)
    
    if result.get("success"):
        return {
            "parsed_file_data": result, 
            # "file_path": None, # Keep file path for later attachment
            "messages": [AIMessage(content=f"已成功解析文件：{os.path.basename(file_path)}。正在为您准备发布草稿...")]
        }
    else:
        return {
            "file_path": None, # Clear if failed
            "messages": [AIMessage(content=f"文件解析失败：{result.get('error')}")]
        }

from langchain_core.runnables import RunnableConfig

def publisher_bridge_node(state: PublisherMasterState, config: RunnableConfig):
    """
    Bridge to Publisher Agent.
    """
    messages = state["messages"]
    parsed_data = state.get("parsed_file_data", {})
    user_info = state.get("user_info", {})
    
    # Prepare inputs
    publisher_state = state.get("publisher_state") or {} # Default to empty dict if None
    publisher_inputs = {
        "messages": messages,
        "user_id": user_info.get("id") or 557, # Default to 557 if missing
        "org_id": user_info.get("org_id") or 6, # Default to 6 if missing
        "current_draft_id": publisher_state.get("current_draft_id", 0),
        "draft_data": publisher_state.get("draft_data", {}),
        "suggested_tags": {},
        "selected_tags": {},
        "next_step": "",
        "is_complete": False
    }
    
    # Inject Parsed Data if available and first run (no draft data yet)
    if parsed_data and not publisher_inputs["draft_data"]:
        extracted = parsed_data.get("extracted_data", {})
        publisher_inputs["draft_data"] = extracted
        # No longer injecting System Message into conversation history.
        # The publisher_agent now reads 'draft_data' directly from state and inserts it into System Prompt.
        
    elif not parsed_data and not publisher_inputs["draft_data"]:
        # Pure dialog mode - Notify agent to start from scratch
        # We can keep this one or remove it too. 
        # If removed, the agent relies on 'draft_data' being empty in System Prompt to know it's a fresh start.
        pass

    # Run Publisher Agent
    result = publisher_app.invoke(publisher_inputs, config=config)
    
    # Capture output
    final_messages = result["messages"]
    last_msg = final_messages[-1]
    
    # Detect if Requirement was Saved/Published
    new_req_id = 0
    final_req_data = {}
    
    # Scan for Tool Message from 'save_requirement'
    # We look at the last few messages
    for m in reversed(final_messages):
        if isinstance(m, ToolMessage) and m.name == "save_requirement":
            if "ID:" in m.content:
                try:
                    new_req_id = int(m.content.split("ID:")[1].strip())
                    # Retrieve the actual requirement to get full data for vector sync
                    try:
                        req = Requirement.objects.get(id=new_req_id)
                        tags1 = [t.value for t in req.tag1.all()]
                        tags2 = [t.post for t in req.tag2.all()]
                        final_req_data = {
                            "id": req.id,
                            "title": req.title,
                            "description": req.description, # Already contains research/skill appended
                            "tags": tags1 + tags2,
                            "status": req.status
                        }
                    except:
                        pass
                except:
                    pass
            break
            
    # === Handle File Attachment and Cleanup ===
    import shutil
    from project.models import File
    
    file_path = state.get("file_path")
    original_name = state.get("original_filename", "document.pdf")
    
    # Only proceed if we have a valid requirement ID and a file path
    if new_req_id and file_path and os.path.exists(file_path):
        try:
            # 1. Define target path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # IMPORTANT: Ensure this path matches your server configuration (e.g. /home/bupt/Server_Project_ZH/media/...)
            # If running on server where Server_Project_ZH is separate, adjust accordingly or use relative to current script if possible.
            # Assuming 'static' or 'media' dir in project root. User mentioned '/home/bupt/Server_Project_ZH/media/uploads/files/requirements'
            # We will use 'static' as per previous code, but note user input mentioned 'media'.
            # For consistency with previous 'static' usage in this file, we stick to 'static' but add robustness.
            # If user wants media, we should change 'static' to 'media'. 
            # Let's use 'media' as requested in user prompt "Server_Project_ZH/media/uploads/files/requirements" (approx).
            # User said: "/home/bupt/Server_Project_ZH/media/uploads/files/requirements"
            # But previous code used "static". I will use "media" to align with user request 3.
            
            # Adjust to user request: /home/bupt/Server_Project_ZH/media/uploads/files/requirement
            SERVER_PROJECT_ROOT = "/home/bupt/Server_Project_ZH"
            media_root = os.path.join(SERVER_PROJECT_ROOT, "media")
            target_dir = os.path.join(media_root, "uploads", "files", "requirement")
            os.makedirs(target_dir, exist_ok=True)
            
            # 2. Generate new unique filename
            ext = os.path.splitext(original_name)[1]
            new_filename = f"{uuid.uuid4().hex}{ext}"
            target_path = os.path.join(target_dir, new_filename)
            
            # 3. Move file (this also removes it from tmp)
            # shutil.move handles cross-device moves by copy+delete
            shutil.move(file_path, target_path)
            
            # 4. Create File model record
            # URL should correspond to how media is served. Usually /media/...
            file_url = f"/media/uploads/files/requirement/{new_filename}"
            file_size = os.path.getsize(target_path)
            
            # real_path should be relative to MEDIA_ROOT for portability and consistency with Django FileField
            relative_path = f"uploads/files/requirement/{new_filename}"
            
            file_obj = File.objects.create(
                name=original_name,
                url=file_url,
                path=f"/需求附件/{original_name}", # Virtual path
                real_path=relative_path, # Storing relative path
                is_folder=False,
                size=file_size
            )
            
            # 5. Link to Requirement
            req = Requirement.objects.get(id=new_req_id)
            req.files.add(file_obj)
            
            print(f"File attached to Requirement {new_req_id}: {target_path}")
            
        except Exception as e:
            print(f"Error attaching file: {e}")
        finally:
            # Ensure tmp file is cleaned up even if move failed (or if move was copy+delete and failed mid-way)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up tmp file: {file_path}")
                except:
                    pass
    # Removed aggressive cleanup of orphaned files to allow multi-turn conversations
    # elif file_path and os.path.exists(file_path):
    #     ...

    return {
        "messages": [last_msg],
        "publisher_state": result,
        "final_requirement_id": new_req_id,
        "final_requirement_data": final_req_data,
        # "file_path": None, # Removed to persist file path
    }

def vector_sync_node(state: PublisherMasterState):
    """
    Sync vectors to Milvus (Doc Chunks + Semantic Embedding).
    Note: Since the backend signals (post_save/m2m_changed) already trigger synchronization,
    this node is primarily for waiting/verification or legacy fallback.
    We skip manual synchronization to avoid race conditions and duplicate data.
    """
    req_id = state.get("final_requirement_id")
    if not req_id:
        return {}
        
    log_msgs = ["已触发后台向量同步。"]
    
    # We rely on Django Signals (project/signals.py) to handle the sync.
    # When 'save_requirement' was called, post_save triggered sync_requirement_vectors (Semantic).
    # When 'req.files.add(file_obj)' was called above, m2m_changed triggered sync_raw_docs_auto (Raw Docs).
    
    return {
        # "final_requirement_id": None, # Keep for caller/test
        # "final_requirement_data": None,
        "parsed_file_data": None, 
        "messages": [AIMessage(content=f"[系统] 需求处理完成。{' '.join(log_msgs)}")]
    }

# from langgraph.checkpoint.memory import MemorySaver

# --- 3. Build Graph ---
workflow = StateGraph(PublisherMasterState)

workflow.add_node("router", router_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("file_parsing_flow", file_parsing_node)
workflow.add_node("publisher_flow", publisher_bridge_node)
workflow.add_node("vector_sync", vector_sync_node)

workflow.set_entry_point("router")

def route_decision(state: PublisherMasterState):
    return state["next_step"]

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "chat_node": "chat_node",
        "file_parsing_flow": "file_parsing_flow",
        "publisher_flow": "publisher_flow"
    }
)

workflow.add_edge("chat_node", END)
workflow.add_edge("file_parsing_flow", "publisher_flow")

# Check if we need to sync vectors or end
def check_sync(state: PublisherMasterState):
    if state.get("final_requirement_id"):
        return "vector_sync"
    return END

workflow.add_conditional_edges("publisher_flow", check_sync, ["vector_sync", END])
workflow.add_edge("vector_sync", END)

# Setup Redis Checkpointer
redis_host = Config.REDIS_HOST
redis_port = Config.REDIS_PORT
redis_db = Config.REDIS_DB

try:
    conn = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    checkpointer = PickleRedisSaver(client=conn)
except Exception as e:
    print(f"Warning: Failed to initialize Redis checkpointer for Publisher: {e}. Fallback to MemorySaver.")
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()

publisher_main_app = workflow.compile(checkpointer=checkpointer)
