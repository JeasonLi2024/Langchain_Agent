
import json
import sys
import os
import django

# Add parent directory to sys.path to resolve relative imports when running as script/module
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_root = os.path.dirname(current_dir) # langchain-v2.0
sys.path.append(langchain_root)

# Initialize Django via core.django_setup
from core.django_setup import setup_django
setup_django()

from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from core.config import Config
from core.prompts import (
    MAIN_ROUTER_PROMPT, 
    MAIN_CHAT_SYSTEM_MESSAGE, 
    RECOMMENDATION_SUMMARY_PROMPT,
    PROJECT_QA_PROMPT
)
from graph.student_workflow import app as recommendation_graph
from tools.new_search_tools import retrieve_project_chunks

# --- 1. Define Master State ---
class MasterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: str
    user_info: dict  # Placeholder for user session info
    user_profile: dict # Short-term memory for user profile (tags, recommendations)
    
    # QA Context
    target_project_id: int
    
    # Subgraph compatibility fields (AgentState)
    user_input: str
    student_id: int
    profile_data: Dict[str, Any]
    final_output: str
    keywords: Any
    context_str: str
    projects_json: Any
    # New RAG fields
    interest_ids: List[int]
    skill_ids: List[int]
    raw_candidates: Dict[int, Dict]
    ranked_projects: List[Dict]

# --- 2. Define Nodes ---

def router_node(state: MasterState):
    """
    Main Agent (Router).
    Analyzes user intent and routes to:
    - 'recommendation_flow': New recommendations.
    - 'project_qa_flow': Specific project questions.
    - 'chat_response': General chat.
    """
    messages = state["messages"]
    last_user_msg = messages[-1].content
    user_profile = state.get("user_profile", {})
    has_recommendations = "Yes" if user_profile else "No"
    
    # Enhanced Intent Classification
    llm = Config.get_utility_llm()
    
    # Format recommendation context for the LLM
    recommendation_context = ""
    if user_profile and "recommended_projects" in user_profile:
        projects = user_profile["recommended_projects"]
        recommendation_context = "User has received these recommendations:\n"
        for i, p in enumerate(projects):
            recommendation_context += f"{i+1}. [ID: {p.get('id')}] {p.get('title')}\n"
    
    prompt = MAIN_ROUTER_PROMPT
    
    try:
        response = llm.invoke(prompt.format(
            message=last_user_msg, 
            has_recommendations=has_recommendations,
            recommendation_context=recommendation_context
        ))
        # Parse JSON output (assuming LLM returns JSON or we parse it)
        content = response.content.strip()
        # Heuristic parsing if LLM is chatty
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            # Fallback text parsing
            if "RECOMMEND" in content: result = {"intent": "RECOMMEND"}
            elif "PROJECT_QA" in content: result = {"intent": "PROJECT_QA", "target_id": 0} # Need ID extraction logic
            else: result = {"intent": "CHAT"}
            
        intent = result.get("intent", "CHAT").upper()
        target_id = result.get("target_id", 0)
        
        # Simple regex fallback for ID if LLM missed it
        if intent == "PROJECT_QA" and not target_id:
            id_match = re.search(r'(\d{3,})', last_user_msg) # Look for 3+ digit numbers
            if id_match:
                target_id = int(id_match.group(1))
        
        if intent == "RECOMMEND":
            return {"next_step": "recommendation_flow"}
        elif intent == "PROJECT_QA" and target_id:
            return {"next_step": "project_qa_flow", "target_project_id": target_id}
        else:
            return {"next_step": "chat_response"}
    except:
        return {"next_step": "chat_response"}

def chat_node(state: MasterState):
    """General Chat Node."""
    llm = Config.get_utility_llm()
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    
    system_message = MAIN_CHAT_SYSTEM_MESSAGE

    if user_profile:
        profile_str = json.dumps(user_profile, ensure_ascii=False)
        system_message += f"\n\n当前用户的推荐结果数据（短期记忆）：\n{profile_str}\n\n如果用户询问关于推荐项目的细节，请参考上述数据回答。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}

def prep_recommendation_node(state: MasterState):
    """Prepare input for recommendation subgraph."""
    user_info = state.get("user_info", {})
    return {
        "user_input": state["messages"][-1].content,
        "student_id": user_info.get("id") or 700,
        "profile_data": {} 
    }

def summarize_recommendation_node(state: MasterState):
    """Summarize the result from the subgraph."""
    profile_data = state.get("profile_data", {})
    user_input = state.get("user_input", "")
    
    if not profile_data and state.get("final_output"):
        import re
        content = state["final_output"]
        try:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
            if json_match:
                profile_data = json.loads(json_match.group(1))
        except:
            pass
    
    if not profile_data:
        return {"messages": [AIMessage(content="抱歉，我没有找到合适的推荐结果。")]}
        
    llm = Config.get_utility_llm()
    summary_prompt = RECOMMENDATION_SUMMARY_PROMPT
    
    chain = ChatPromptTemplate.from_template(summary_prompt) | llm
    response = chain.invoke({
        "user_input": user_input,
        "json_data": json.dumps(profile_data, ensure_ascii=False)
    })
    
    return {
        "messages": [response],
        "user_profile": profile_data
    }

def project_qa_node(state: MasterState):
    """
    QA Node for specific projects.
    Retrieves chunks from Milvus and answers questions.
    """
    target_id = state.get("target_project_id")
    messages = state["messages"]
    last_question = messages[-1].content
    
    # 1. Retrieve Context
    context_chunks = retrieve_project_chunks.invoke({
        "project_ids": [target_id],
        "query": last_question
    })
    
    # context_chunks is a dict: {project_id: [chunks]}
    chunks_list = context_chunks.get(target_id, []) if context_chunks else []
    context_text = "\n\n".join(chunks_list)
    
    if not context_text:
        context_text = "No specific documents found for this project."
        
    # 2. Generate Answer
    llm = Config.get_utility_llm()
    qa_prompt = PROJECT_QA_PROMPT
    
    chain = ChatPromptTemplate.from_template(qa_prompt) | llm
    response = chain.invoke({
        "target_id": target_id,
        "context": context_text,
        "question": last_question
    })
    
    return {"messages": [response]}

# --- 3. Build Master Graph ---
workflow = StateGraph(MasterState)

workflow.add_node("router", router_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("prep_recommendation", prep_recommendation_node)
workflow.add_node("recommendation_worker", recommendation_graph)
workflow.add_node("summarize_recommendation", summarize_recommendation_node)
workflow.add_node("project_qa_node", project_qa_node)

workflow.set_entry_point("router")

def route_decision(state: MasterState):
    return state["next_step"]

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "chat_response": "chat_node",
        "recommendation_flow": "prep_recommendation",
        "project_qa_flow": "project_qa_node"
    }
)

workflow.add_edge("chat_node", END)
workflow.add_edge("prep_recommendation", "recommendation_worker")
workflow.add_edge("recommendation_worker", "summarize_recommendation")
workflow.add_edge("summarize_recommendation", END)
workflow.add_edge("project_qa_node", END)

master_app = workflow.compile()
