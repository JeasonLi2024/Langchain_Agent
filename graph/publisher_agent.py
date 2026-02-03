import json
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
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
    draft_id: int = 0
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
    """
    try:
        # Check user and org
        user = User.objects.get(id=user_id)
        org_user = OrganizationUser.objects.get(user_id=user_id, organization_id=org_id)
        
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

        req.save()
        
        # Update Tags
        if tag1_ids:
            req.tag1.set(tag1_ids)
        if tag2_ids:
            req.tag2.set(tag2_ids)
            
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

# --- Nodes ---

def chat_node(state: PublisherState):
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
    
    # Bind tools (Both save_requirement and recommend_tags)
    llm_with_tools = llm.bind_tools([save_requirement, recommend_tags])
    
    # Invoke
    response = llm_with_tools.invoke([SystemMessage(content=system_msg)] + messages)
    
    return {"messages": [response]}

def tag_recommendation_node(state: PublisherState):
    """
    Node to handle tag recommendation logic.
    Acts as the execution of 'recommend_tags' tool.
    """
    draft_data = state.get("draft_data", {})
    description = draft_data.get("description", "")
    research_direction = draft_data.get("research_direction", "")
    skill = draft_data.get("skill", "")
    
    # Call the logic
    result_text = recommend_tags_logic(description, research_direction, skill)
    
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
        # Default to standard tools node for other tools (save_requirement)
        return "tools"
        
    return END

workflow.add_conditional_edges("chat", should_continue, ["tools", "tag_recommendation", END])
workflow.add_edge("tools", "chat")
workflow.add_edge("tag_recommendation", "chat") # Loop back to agent to summarize

# Compile with interrupt_before for HITL on sensitive tool usage
# NOTE: In production with custom UI, you must handle the 'interrupt' state.
# For testing with basic UI, we disable it to prevent freezing.
# publisher_app = workflow.compile(interrupt_before=["tools"])
publisher_app = workflow.compile()
