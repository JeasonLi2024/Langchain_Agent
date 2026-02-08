import json
import sys
import os
import django
import base64
import tempfile
import uuid

import re
import traceback

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prioritize Local Env
from core.config import Config

# Initialize Django via core.django_setup
from core.django_setup import setup_django
setup_django()

from typing_extensions import TypedDict, Annotated, List, Literal, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
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
from project.signals import handle_requirement_save, handle_files_change, handle_tags_change
from django.db.models.signals import post_save, m2m_changed
from contextlib import contextmanager

@contextmanager
def suppress_signals():
    """
    Temporarily disconnect Django signals to prevent automatic Celery tasks 
    from overwriting our manual vector sync.
    """
    try:
        post_save.disconnect(handle_requirement_save, sender=Requirement)
        m2m_changed.disconnect(handle_files_change, sender=Requirement.files.through)
        m2m_changed.disconnect(handle_tags_change, sender=Requirement.tag1.through)
        m2m_changed.disconnect(handle_tags_change, sender=Requirement.tag2.through)
        yield
    finally:
        post_save.connect(handle_requirement_save, sender=Requirement)
        m2m_changed.connect(handle_files_change, sender=Requirement.files.through)
        m2m_changed.connect(handle_tags_change, sender=Requirement.tag1.through)
        m2m_changed.connect(handle_tags_change, sender=Requirement.tag2.through)

# Import Postgres Saver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from core.db import PostgresPool

# --- 1. Define State ---

class PublisherInputState(TypedDict):
    """
    Input schema for the graph.
    Only allows messages and user_info to be passed in, preventing
    accidental overwrites of internal state (file_path, parsed_data) 
    by clients sending empty defaults.
    """
    messages: List[BaseMessage]
    user_info: Optional[dict]

class PublisherMasterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: str
    user_info: dict
    
    # Publisher Context
    file_path: Optional[str] # If user uploaded a file
    cover_image_path: Optional[str] # If user uploaded a cover image
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
                            elif 'image' in mime_type:
                                if 'png' in mime_type:
                                    ext = ".png"
                                elif 'jpeg' in mime_type or 'jpg' in mime_type:
                                    ext = ".jpg"
                                else:
                                    ext = ".jpg" # Default image
                                
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
                            
                            # Check if it is an image
                            is_image = False
                            lower_name = original_filename.lower()
                            if lower_name.endswith('.png') or lower_name.endswith('.jpg') or lower_name.endswith('.jpeg') or 'image' in mime_type:
                                is_image = True
                                
                            # Handle Image Persistence to MEDIA_ROOT/cover/tmp immediately
                            if is_image:
                                try:
                                    from django.conf import settings
                                    import shutil
                                    
                                    # Move to cover/tmp (Web accessible)
                                    cover_tmp_dir = os.path.join(settings.MEDIA_ROOT, "cover", "tmp")
                                    os.makedirs(cover_tmp_dir, exist_ok=True)
                                    
                                    # Use safe filename already generated
                                    target_filename = os.path.basename(file_path)
                                    target_path = os.path.join(cover_tmp_dir, target_filename)
                                    
                                    # Move from generic tmp to specific cover tmp
                                    shutil.move(file_path, target_path)
                                    file_path = target_path # Update to new location
                                    
                                    # Build Full Web URL
                                    relative_path = f"cover/tmp/{target_filename}"
                                    base_domain = "https://zhihui.bupt.edu.cn"
                                    media_prefix = settings.MEDIA_URL
                                    if not media_prefix.startswith("/"): media_prefix = "/" + media_prefix
                                    if not media_prefix.endswith("/"): media_prefix = media_prefix + "/"
                                    
                                    cover_url = f"{base_domain}{media_prefix}{relative_path}"
                                    print(f"Image moved to {target_path}, URL: {cover_url}")
                                except Exception as e:
                                    print(f"Error processing image upload: {e}")
                            
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
    
    # Handle Image Uploads specifically (Bypass parsing, inject system info)
    if new_file_uploaded and is_image:
        # Inject a system message to inform the agent about the file
        # We append it to the messages list in the state update (returned dict)
        # But we can't easily modify 'messages' here without returning it.
        # So we return it in the result.
        
        # We need to construct a message that the agent can see.
        # Since we are in a router, we return state updates.
        img_ref = cover_url if cover_url else file_path
        sys_msg = SystemMessage(content=f"User uploaded an image file: {original_filename}. URL/Path: {img_ref}. Use this if the user wants to set it as cover.")
        
        # Check if we are in publisher flow
        publisher_state = state.get("publisher_state", {})
        if publisher_state and not publisher_state.get("is_complete", False):
             return {
                 "next_step": "publisher_flow", 
                 "messages": [sys_msg],
                 "cover_image_path": img_ref 
             }
        else:
             # Default to chat or publisher based on intent, but skip parsing
             # If it's just an image, maybe we start publisher flow?
             return {
                 "next_step": "publisher_flow",
                 "messages": [sys_msg],
                 "cover_image_path": img_ref
             }

    # Priority 1: File Upload -> Parsing (Only for non-images)
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
    
    # Sanitize messages to ensure content is string for ChatTongyi (doesn't support list/multimodal content)
    sanitized_messages = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            # Extract text parts from multimodal content
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'image_url' or item.get('type') == 'file':
                        text_parts.append("[User uploaded an image]")
            content = " ".join(text_parts)
            
        # Reconstruct message with string content
        if isinstance(m, HumanMessage):
            sanitized_messages.append(HumanMessage(content=content))
        elif isinstance(m, AIMessage):
            sanitized_messages.append(AIMessage(content=content))
        elif isinstance(m, SystemMessage):
            sanitized_messages.append(SystemMessage(content=content))
        elif isinstance(m, ToolMessage):
            sanitized_messages.append(ToolMessage(content=content, tool_call_id=m.tool_call_id, name=m.name))
        else:
            # Fallback for other types
            sanitized_messages.append(HumanMessage(content=str(content)))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PUBLISHER_CHAT_SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": sanitized_messages})
    
    return {"messages": [response]}

async def file_parsing_node(state: PublisherMasterState):
    """
    Handle file parsing.
    """
    file_path = state["file_path"]
    original_filename = state.get("original_filename") or os.path.basename(file_path)
    
    input_state = {
        "file_path": file_path,
        "file_name": original_filename,
    }
    
    result = await file_parsing_app.ainvoke(input_state)
    
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

from asgiref.sync import sync_to_async

async def publisher_bridge_node(state: PublisherMasterState, config: RunnableConfig):
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
        "is_complete": False,
        "cover_image_path": state.get("cover_image_path") # Pass cover image path
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
    # Suppress signals to prevent automatic vector sync from interfering with manual sync
    with suppress_signals():
        result = await publisher_app.ainvoke(publisher_inputs, config=config)
    
    # Capture output
    final_messages = result["messages"]
    
    # Calculate new messages to return to Master Graph
    # We filter out messages that were already in the input
    input_len = len(messages)
    new_messages = final_messages[input_len:]
    
    # Fallback: If for some reason new_messages is empty but result has messages (e.g. logic change),
    # ensure we return at least the last one.
    if not new_messages and final_messages:
        new_messages = [final_messages[-1]]
        
    last_msg = final_messages[-1]
    
    # Detect if Requirement was Saved/Published
    new_req_id = 0
    final_req_data = {}
    
    # Scan for Tool Message from 'save_requirement'
    # We look at the last few messages
    for m in reversed(final_messages):
        if isinstance(m, ToolMessage) and m.name == "save_requirement":
            # Robust ID extraction using regex
            # Matches "ID: 123" or "ID:123" or "ID: 123."
            match = re.search(r"ID:\s*(\d+)", m.content)
            if match:
                try:
                    new_req_id = int(match.group(1))
                    # Retrieve the actual requirement to get full data for vector sync
                    try:
                        # Wrap sync DB call
                        @sync_to_async
                        def get_req_details(req_id):
                            req = Requirement.objects.get(id=req_id)
                            tags1 = [t.value for t in req.tag1.all()]
                            tags2 = [t.post for t in req.tag2.all()]
                            return {
                                "id": req.id,
                                "title": req.title,
                                "description": req.description,
                                "tags": tags1 + tags2,
                                "status": req.status
                            }
                        
                        final_req_data = await get_req_details(new_req_id)
                    except Exception as e:
                        print(f"Error retrieving requirement details: {e}")
                except Exception as e:
                    print(f"Error parsing requirement ID: {e}")
            break
            
    # === Handle File Attachment and Cleanup ===
    import shutil
    from project.models import File
    from project.services import delete_requirement_vectors, get_or_create_collection, COLLECTION_RAW_DOCS, COLLECTION_EMBEDDINGS, generate_embedding
    
    file_path = state.get("file_path")
    original_name = state.get("original_filename", "document.pdf")
    
    # Debug info
    if file_path:
        print(f"Checking file attachment: ReqID={new_req_id}, File={file_path}, Exists={os.path.exists(file_path)}")
    
    # Only proceed if we have a valid requirement ID and a file path
    if new_req_id and file_path and os.path.exists(file_path):
        try:
            # 1. Define target path using Django MEDIA_ROOT
            from django.conf import settings
            
            # 2. Generate new unique filename
            ext = os.path.splitext(original_name)[1]
            new_filename = f"{uuid.uuid4().hex}{ext}"
            
            # Construct relative path (e.g., "files/requirement/uuid.pdf")
            relative_path = f"uploads/files/requirement/{new_filename}"
            
            # Absolute path in Django's Media directory
            target_path = os.path.join(settings.MEDIA_ROOT, relative_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 3. Move file
            shutil.move(file_path, target_path)
            
            # 4. Create File Object
            # Construct URL relative to MEDIA_URL
            file_url = f"{settings.MEDIA_URL}{relative_path}"
            file_size = os.path.getsize(target_path)
            
            # real_path should be relative to MEDIA_ROOT for portability and consistency with Django FileField
            relative_path = f"uploads/files/requirement/{new_filename}"
            
            # Wrap sync DB creation
            @sync_to_async
            def create_and_link_file(req_id, name, url, path, real_path, size):
                # Suppress signals here too
                with suppress_signals():
                    file_obj = File.objects.create(
                        name=name,
                        url=url,
                        path=path,
                        real_path=real_path,
                        is_folder=False,
                        size=size
                    )
                    req = Requirement.objects.get(id=req_id)
                    req.files.add(file_obj)
                    return file_obj

            await create_and_link_file(
                new_req_id,
                original_name,
                file_url,
                f"/需求附件/{original_name}",
                relative_path,
                file_size
            )
            
            print(f"File attached to Requirement {new_req_id}: {target_path}")
            
            # === Manual Vector Sync (Agent-Driven Persistence) ===
            # Ensure consistency by using the EXACT chunks parsed by the Agent
            if parsed_data:
                chunks = parsed_data.get("chunks", [])
                embeddings = parsed_data.get("chunk_embeddings", [])
                print(f"Executing Agent-Driven Vector Sync for Req {new_req_id}...")
                print(f"Parsed Data: Chunks={len(chunks)}, Embeddings={len(embeddings)}")
                
                # 1. Sync Raw Docs (Chunks)
                if chunks and embeddings and len(chunks) == len(embeddings):
                    try:
                        # Clear potential backend fallback data first
                        delete_requirement_vectors(new_req_id, [COLLECTION_RAW_DOCS])
                        
                        collection = get_or_create_collection(COLLECTION_RAW_DOCS)
                        pids = [new_req_id] * len(chunks)
                        # Filter valid embeddings
                        valid_data = [(p, v, c, i) for i, (p, v, c) in enumerate(zip(pids, embeddings, chunks)) if v and len(v) > 0]
                        
                        if valid_data:
                            data_insert = [
                                [x[0] for x in valid_data], # pids
                                [x[1] for x in valid_data], # vectors
                                [x[2][:65535] for x in valid_data], # content
                                [x[3] for x in valid_data]  # indices
                            ]
                            collection.insert(data_insert)
                            collection.flush()
                            print(f"Successfully inserted {len(valid_data)} chunks into {COLLECTION_RAW_DOCS}")
                    except Exception as e:
                        print(f"Error in manual raw docs sync: {e}")
                
                # 2. Sync Semantic Embedding (Project)
                # Use data from final_req_data (which has Tags)
                if final_req_data:
                    try:
                        title = final_req_data.get("title", "")
                        desc = final_req_data.get("description", "")
                        tags = final_req_data.get("tags", [])
                        # Brief might be missing in final_req_data dict, retrieve from draft_data or empty
                        brief = publisher_state.get("draft_data", {}).get("brief", "")
                        
                        full_text = f"Title: {title}\nBrief: {brief}\nDescription: {desc}\nTags: {', '.join(tags)}"
                        
                        vector = generate_embedding(full_text)
                        
                        if vector:
                            delete_requirement_vectors(new_req_id, [COLLECTION_EMBEDDINGS])
                            collection_emb = get_or_create_collection(COLLECTION_EMBEDDINGS)
                            data_emb = [
                                [new_req_id],
                                [vector],
                                [full_text[:65535]]
                            ]
                            collection_emb.insert(data_emb)
                            collection_emb.flush()
                            print(f"Successfully inserted semantic embedding into {COLLECTION_EMBEDDINGS}")
                    except Exception as e:
                        print(f"Error in manual semantic sync: {e}")
                        # If schema mismatch, try to recover (in dev/test env only)
                        if "schema" in str(e).lower() or "match" in str(e).lower():
                             print("Attempting to fix schema mismatch by dropping collection...")
                             try:
                                 from pymilvus import utility
                                 utility.drop_collection(COLLECTION_EMBEDDINGS)
                                 print(f"Dropped {COLLECTION_EMBEDDINGS}, retrying insert...")
                                 collection_emb = get_or_create_collection(COLLECTION_EMBEDDINGS)
                                 collection_emb.insert(data_emb)
                                 collection_emb.flush()
                                 print(f"Retry successful.")
                             except Exception as retry_e:
                                 print(f"Retry failed: {retry_e}")

        except Exception as e:
            print(f"Error attaching file: {e}")
            traceback.print_exc()
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

    # Check for stale files cleanup on each run (simple implementation)
    cleanup_stale_files()

    return {
        "messages": new_messages,
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
workflow = StateGraph(PublisherMasterState, input=PublisherInputState)

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

# Setup Postgres Checkpointer
# We use the global connection pool. 
# Note: The pool must be opened (await pool.open()) by the server lifespan or script before use.
# For LangGraph API compatibility, we do not set checkpointer here.
# It will be injected by server.py (for local prod) or LangGraph Platform (for dev).
publisher_main_app = workflow.compile()
