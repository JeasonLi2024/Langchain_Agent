import json
from typing_extensions import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from core.config import Config
from core.prompts import PUBLISHER_AGENT_SYSTEM_PROMPT
from project.models import Requirement
from user.models import OrganizationUser, User, Tag1, Tag2
from graph.tag_recommendation import recommend_tags_logic

# --- State ---
class PublisherState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: int
    org_id: int
    
    # Context
    current_draft_id: int # If editing existing draft
    draft_data: Dict[str, Any] # Current draft data
    
    # Tagging
    tag_recommendation_trigger: bool # Flag to trigger tag recommendation node
    suggested_tags: Dict[str, Any] # Tags suggested by LLM
    selected_tags: Dict[str, List[int]] # Tags selected by user

    # Cover Image
    cover_flow_status: Literal['pending', 'generating', 'selecting', 'completed'] # Cover image flow status
    cover_image_candidates: List[str] # List of AI generated image URLs
    selected_cover_image: Optional[str] # Final selected image URL (or local path)
    is_local_upload: bool # Whether user chose local upload
    cover_image_path: Optional[str] # Path to user uploaded cover image (passed from router)
    
    # Flags
    next_step: str
    is_complete: bool

@tool
def save_requirement(
    user_id: int, 
    org_id: int, 
    title: str, 
    description: str, 
    brief: str = "", 
    research_direction: str = "",
    skill: str = "",
    finish_time: str = None,
    budget: str = "",
    support_provided: str = "",
    status: Literal['draft', 'under_review'] = 'draft',
    tag1_ids: List[int] = [],
    tag2_ids: List[int] = [],
    draft_id: int = 0,
    cover_image_url: str = None
):
    """
    Save or publish a requirement.
    
    Args:
        user_id: ID of the user.
        org_id: ID of the organization.
        title: Project title.
        description: Detailed description.
        brief: Short introduction (optional).
        research_direction: Research direction keywords.
        skill: Technology stack keywords.
        finish_time: Deadline (YYYY-MM-DD).
        budget: Budget support value in Ten Thousand Yuan (e.g., "50"). Only the number.
        support_provided: Other support.
        status: 'draft' for saving as draft, 'under_review' for publishing.
        tag1_ids: List of Interest Tag IDs.
        tag2_ids: List of Skill Tag IDs.
        draft_id: Existing draft ID to update (0 for new).
        cover_image_url: The URL or path of the selected cover image (optional).
    """
    try:
        # Check user and org
        user = User.objects.get(id=user_id)
        try:
            # Use user object directly as per best practice and to avoid confusion
            org_user = OrganizationUser.objects.get(user=user, organization_id=org_id)
        except OrganizationUser.DoesNotExist:
            return f"Error: User {user_id} is not a member of Organization {org_id}. Please verify the user belongs to this organization."
        
        if draft_id and draft_id > 0:
            try:
                req = Requirement.objects.get(id=draft_id)
            except Requirement.DoesNotExist:
                req = None
        else:
            req = None
            
        if not req:
            req = Requirement(
                organization_id=org_id,
                publish_people=org_user
            )
        
        # Update fields
        req.title = title
        req.description = description
        req.brief = brief or description[:100]
        req.status = status

        
        # Update standard fields present in model
        if finish_time and finish_time != "无":
            req.finish_time = finish_time
        if budget and budget != "无":
            # Ensure only number is stored
            budget_clean = str(budget).replace("万元", "").replace("元", "").strip()
            req.budget = budget_clean
        if support_provided and support_provided != "无":
            req.support_provided = support_provided
            
        # For research_direction and skill, since they are not in model, we append to description if not already there
        # to ensure we don't lose this info.
        extra_info = ""
        if research_direction and research_direction not in description:
            extra_info += f"\n\n【研究方向】\n{research_direction}"
        if skill and skill not in description:
            extra_info += f"\n\n【技术栈】\n{skill}"
            
        if extra_info:
            req.description += extra_info

        # Update Tags
        # NOTE: Signals are suppressed by the parent graph (publisher_main_agent.py).
        # We must handle vector synchronization manually here to ensure data consistency.
        # This applies to both file uploads (handled in parent) and chat-only updates (handled here).
        
        # Handle Cover Image Persistence
        # Check if cover_image_url indicates a temporary file (either in cover/tmp or generic tmp)
        import os  # Import os here to avoid UnboundLocalError
        if cover_image_url and ("cover/tmp/" in cover_image_url or "/tmp/" in cover_image_url or "upload_" in os.path.basename(cover_image_url)):
            try:
                import shutil
                from django.conf import settings
                from datetime import datetime
                
                # 1. Parse Paths
                source_path = None
                # Check if it's an absolute path first
                if os.path.isabs(cover_image_url) and os.path.exists(cover_image_url):
                    source_path = cover_image_url
                else:
                    # Try relative to MEDIA_ROOT (handling URL format)
                    # Handle full URL by stripping domain/scheme if present
                    temp_url = cover_image_url
                    if settings.MEDIA_URL in temp_url:
                         # Robustly strip everything before MEDIA_URL
                         # e.g. https://domain.com/media/foo.png -> foo.png
                         parts = temp_url.split(settings.MEDIA_URL)
                         if len(parts) > 1:
                             relative_path = parts[-1]
                         else:
                             relative_path = temp_url.replace(settings.MEDIA_URL, "")
                    else:
                        relative_path = temp_url.replace(settings.MEDIA_URL, "")

                    if relative_path.startswith("/"):
                        relative_path = relative_path[1:]
                    
                    # Remove query params if any
                    relative_path = relative_path.split('?')[0]
                    
                    potential_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                    if os.path.exists(potential_path):
                        source_path = potential_path
                
                if source_path:
                    # 2. Determine Target Path
                    # Format: cover/%Y/%m/filename
                    now = datetime.now()
                    target_dir_rel = f"cover/{now.year}/{now.month:02d}"
                    target_dir = os.path.join(settings.MEDIA_ROOT, target_dir_rel)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    filename = os.path.basename(source_path)
                    # Ensure unique filename if collision
                    if os.path.exists(os.path.join(target_dir, filename)):
                         name, ext = os.path.splitext(filename)
                         filename = f"{name}_{int(now.timestamp())}{ext}"

                    target_path = os.path.join(target_dir, filename)
                    
                    # 3. Move File
                    shutil.move(source_path, target_path)
                    
                    # 4. Update Model Field (Save relative path to MEDIA_ROOT)
                    # Django ImageField stores relative path
                    req.cover = f"{target_dir_rel}/{filename}"
                    
                    # 5. Cleanup Stale Images (Only for AI generated batches)
                    # File format: {batch_id}_{idx}.png
                    try:
                        if "cover/tmp/" in source_path:
                            batch_id = filename.split('_')[0]
                            tmp_dir = os.path.dirname(source_path) # cover/tmp
                            
                            for f in os.listdir(tmp_dir):
                                if f.startswith(batch_id + '_'):
                                    try:
                                        os.remove(os.path.join(tmp_dir, f))
                                    except OSError:
                                        pass
                    except Exception as e:
                        print(f"Warning: Cleanup failed: {e}")
                        
                else:
                    print(f"Warning: Source cover image not found: {cover_image_url}")
                    
            except Exception as e:
                    print(f"Error processing cover image: {e}")
        elif cover_image_url:
            # If it's not a tmp file, just save it (assuming it's a valid URL or path)
            # But strictly speaking we should probably ensure it's relative to MEDIA_ROOT if it's a local file
            req.cover = cover_image_url

        req.save()
        
        if tag1_ids:
            req.tag1.set(tag1_ids)
        if tag2_ids:
            req.tag2.set(tag2_ids)
            
        # Manual Vector Sync Trigger
        # Ensure vectorization happens even if Celery tasks are not running or fail
        if status == 'under_review':
            try:
                from project.services import sync_requirement_vectors, sync_raw_docs_auto
                print(f"Triggering manual vector sync for Requirement {req.id}...")
                sync_requirement_vectors(req)
                sync_raw_docs_auto(req)
                print(f"Manual vector sync completed for Requirement {req.id}.")
            except Exception as e:
                print(f"Error in manual vector sync: {e}")

        action = "Published" if status == 'under_review' else "Draft Saved"
        return f"{action} successfully. ID: {req.id}"
        
    except Exception as e:
        return f"Error saving requirement: {str(e)}"

@tool
def recommend_tags():
    """
    Trigger the tag recommendation process. 
    Use this tool when the user confirms they want tag recommendations or when you determine it's time to recommend tags.
    """
    return "Starting tag recommendation..."

@tool
def generate_cover_images(style: str = "default"):
    """
    Generate AI cover images for the requirement.
    
    Args:
        style: The style of the image. Options: 'default', 'tech', 'illustration', 'ink', '3d'. Default is 'default'.
    """
    return f"Starting AI cover image generation with style: {style}"

@tool
def select_cover_image(image_url: str):
    """
    Select a specific cover image (either AI generated or locally uploaded).
    
    Args:
        image_url: The URL or path of the selected image.
    """
    return f"Cover image selected: {image_url}"

from langchain_core.runnables import RunnableConfig

# --- Nodes ---

async def cover_flow_node(state: PublisherState, config: RunnableConfig):
    """
    Node to handle cover image generation logic.
    Executes actual API calls when tools are invoked.
    """
    messages = state['messages']
    last_message = messages[-1]
    
    # Initialize return state
    new_state = {}
    
    # Check if this node was triggered by a tool call
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'generate_cover_images':
                # Execute AI generation
                from tools.ai_utils import generate_poster_images
                
                draft_data = state.get('draft_data', {})
                title = draft_data.get('title', 'Project Requirement')
                brief = draft_data.get('brief', '')
                if not brief:
                    brief = draft_data.get('description', '')[:200]
                
                # Get tags from state
                tags = []
                suggested = state.get('suggested_tags', {})
                selected_tag_ids = state.get('selected_tags', {})
                
                # Try to resolve tag names (simplified logic, ideally map IDs back to names)
                # For now just use title/brief context mostly, or use raw tag strings if available
                # If we have suggested tags names, use them
                if suggested and 'interest_tags' in suggested:
                     tags.extend([t['name'] for t in suggested['interest_tags']])
                
                style = tool_call['args'].get('style', 'default')
                
                try:
                    # We need a dummy request object or handle URL building without it if possible
                    # ai_utils.py uses build_media_url which might need request. 
                    # Assuming basic settings.MEDIA_URL works.
                    image_urls = generate_poster_images(title, brief, tags, style=style)
                    
                    new_state['cover_image_candidates'] = image_urls
                    new_state['cover_flow_status'] = 'selecting'
                    
                    # Return tool output
                    return {
                        "messages": [ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=json.dumps({
                                "status": "success", 
                                "images": image_urls,
                                "message": "已为您生成4张封面图，请选择一张。"
                            }, ensure_ascii=False)
                        )],
                        **new_state
                    }
                except Exception as e:
                     return {
                        "messages": [ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=f"Error generating images: {str(e)}"
                        )]
                    }

            elif tool_call['name'] == 'select_cover_image':
                image_url = tool_call['args'].get('image_url')
                new_state['selected_cover_image'] = image_url
                new_state['cover_flow_status'] = 'completed'
                
                return {
                    "messages": [ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=f"已选择封面图: {image_url}"
                    )],
                    **new_state
                }

    return {}

async def chat_node(state: PublisherState, config: RunnableConfig):
    """
    Main interaction node.
    """
    llm = Config.get_utility_llm()
    messages = state['messages']
    
    user_id = state.get('user_id')
    org_id = state.get('org_id')
    current_draft_id = state.get('current_draft_id', 0)
    draft_data = state.get('draft_data', {})
    
    # Prepare Draft Context
    draft_context_str = "暂无初始信息"
    if draft_data:
        key_map = {
            "title": "标题",
            "brief": "简介",
            "description": "详细描述",
            "research_direction": "研究方向",
            "skill": "技术栈",
            "finish_time": "完成时间",
            "budget": "预算",
            "support_provided": "可提供的支持"
        }
        items = []
        for k, v in draft_data.items():
            cn_key = key_map.get(k, k)
            display_val = v
            # Add unit for display if it's budget and missing unit
            if k == 'budget' and v and v != '无' and '万元' not in str(v):
                 display_val = f"{v}万元"
            items.append(f"- {cn_key}: {display_val}")
        draft_context_str = "\n".join(items)
    
    # System Prompt
    system_msg = PUBLISHER_AGENT_SYSTEM_PROMPT.format(
        user_id=user_id,
        org_id=org_id,
        current_draft_id=current_draft_id,
        draft_context_str=draft_context_str
    )
    
    # Inject Cover Image Context if available
    cover_image_path = state.get("cover_image_path")
    if cover_image_path:
        system_msg += f"\n\n【重要提示】检测到用户上传了封面图片：{cover_image_path}。\n请务必在回复中向用户展示这张图片（使用Markdown图片语法 `![]({cover_image_path})`），并询问用户是否确认使用该图片作为封面。\n如果用户确认，请在调用 `save_requirement` 时将 `cover_image_url` 参数设置为此路径。"

    # Dynamic Prompt Injection: Force Tool Call if user asks for recommendation
    last_human_msg = ""
    last_ai_msg = ""
    
    # Get last human message
    for m in reversed(messages):
        if m.type == 'human':
            last_human_msg = str(m.content)
            break
            
    # Get last AI message (to check if we asked about tags)
    for m in reversed(messages):
        if m.type == 'ai':
            last_ai_msg = str(m.content)
            break
            
    # Check triggers
    user_asks_directly = "推荐" in last_human_msg or "标签" in last_human_msg
    
    # Check if user confirms a previous question about tags
    # E.g. AI: "需要推荐标签吗？" Human: "是的"
    affirmative_words = ["是", "好", "对", "行", "可以", "没问题", "确认", "需要", "ok", "OK"]
    user_confirms = (
        ("推荐" in last_ai_msg or "标签" in last_ai_msg) and 
        any(word in last_human_msg for word in affirmative_words)
    )

    # Prevent infinite loop: if we just ran the tool, don't force it again
    just_ran_recommend_tags = False
    if messages and isinstance(messages[-1], ToolMessage) and messages[-1].name == 'recommend_tags':
        just_ran_recommend_tags = True

    if (user_asks_directly or user_confirms) and not just_ran_recommend_tags:
        system_msg += "\n\n【重要指令】用户正在请求或同意标签推荐。请务必调用 `recommend_tags` 工具，**不要**直接在文本中回复标签。"
        # Force tool usage if supported by the LLM provider
        # llm_with_tools = llm.bind_tools([save_requirement, recommend_tags], tool_choice="recommend_tags")
        # However, to be safe with different providers, we stick to prompt engineering + auto for now, 
        # unless we are sure about the provider's capability.
        # Let's try forcing it via prompt is usually safer if we don't know the exact tool name mapping.
    
    # Bind tools (Add cover image tools)
    llm_with_tools = llm.bind_tools([save_requirement, recommend_tags, generate_cover_images, select_cover_image])
    
    # Process Message History
    # We must sanitize history to ensure multimodal content is handled or flattened
    sanitized_messages = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
             # Preserve multimodal content for models that support it (e.g. Qwen-VL, GPT-4o)
             new_content = []
             for item in content:
                 if isinstance(item, dict):
                    if item.get('type') == 'text':
                        new_content.append(item)
                    elif item.get('type') == 'image_url':
                        # Preserve image_url
                        new_content.append(item)
                        # Add a system hint so the agent knows this is likely a cover image candidate
                        new_content.append({"type": "text", "text": "[System: User has uploaded this image. You should acknowledge it and ask if they want to use it as the project cover.]"})
                    elif item.get('type') == 'image':
                        # Convert raw image to text placeholder
                        new_content.append({"type": "text", "text": "[System: User has uploaded an image. You should acknowledge it and ask if they want to use it as the project cover.]"})
                    elif item.get('type') == 'file':
                        new_content.append({"type": "text", "text": "[System: User has uploaded a file attachment.]"})
             
             if new_content:
                 content = new_content
             else:
                 content = "" # Handle empty case

        if m.type == 'human':
            sanitized_messages.append(HumanMessage(content=content))
        elif m.type == 'ai':
            sanitized_messages.append(AIMessage(content=content, additional_kwargs=m.additional_kwargs))
        elif m.type == 'system':
            sanitized_messages.append(SystemMessage(content=content))
        elif m.type == 'tool':
            sanitized_messages.append(ToolMessage(content=content, tool_call_id=m.tool_call_id, name=m.name))
        else:
            sanitized_messages.append(HumanMessage(content=str(content)))

    # Invoke
    response = await llm_with_tools.ainvoke([SystemMessage(content=system_msg)] + sanitized_messages, config=config)
    
    return {"messages": [response]}

async def tag_recommendation_node(state: PublisherState, config: RunnableConfig):
    """
    Node to handle tag recommendation logic.
    Acts as the execution of 'recommend_tags' tool.
    """
    draft_data = state.get("draft_data", {})
    description = draft_data.get("description", "")
    research_direction = draft_data.get("research_direction", "")
    skill = draft_data.get("skill", "")
    
    # Call the logic
    result_text = await recommend_tags_logic(description, research_direction, skill, config=config)
    
    # Find the tool call ID to respond to
    last_message = state["messages"][-1]
    tool_call_id = "unknown"
    if last_message.tool_calls:
        # Assuming the last tool call is recommend_tags
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "recommend_tags":
                tool_call_id = tool_call["id"]
                break
    
    # Return a ToolMessage to satisfy the chat history
    # The content contains the "Thinking" + JSON
    tag_msg = ToolMessage(content=result_text, tool_call_id=tool_call_id, name="recommend_tags")
    
    return {
        "messages": [tag_msg],
        "tag_recommendation_trigger": False 
    }

# --- Graph ---
workflow = StateGraph(PublisherState)

workflow.add_node("chat", chat_node)
workflow.add_node("tag_recommendation", tag_recommendation_node)
workflow.add_node("cover_flow", cover_flow_node) # Add cover flow node

from langgraph.prebuilt import ToolNode
tool_node = ToolNode([save_requirement])
workflow.add_node("tools", tool_node)

workflow.set_entry_point("chat")

def should_continue(state: PublisherState):
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        # Check which tool was called
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "recommend_tags":
                return "tag_recommendation"
            elif tool_call["name"] in ["generate_cover_images", "select_cover_image"]:
                return "cover_flow"
        # Default to standard tools node for other tools (save_requirement)
        return "tools"
        
    return END

workflow.add_conditional_edges("chat", should_continue, ["tools", "tag_recommendation", "cover_flow", END])
workflow.add_edge("tools", "chat")
workflow.add_edge("tag_recommendation", "chat") # Loop back to agent to summarize
workflow.add_edge("cover_flow", "chat") # Loop back to agent

# Compile with interrupt_before for HITL on sensitive tool usage
# NOTE: In production with custom UI, you must handle the 'interrupt' state.
# For testing with basic UI, we disable it to prevent freezing.
# publisher_app = workflow.compile(interrupt_before=["tools"])
publisher_app = workflow.compile()
