
import json
import re
from typing_extensions import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.prompts import REASONING_GEN_SYSTEM_PROMPT, REASONING_GEN_HUMAN_PROMPT
from tools.search_tools import extract_keywords, retrieve_tags
from tools.new_search_tools import search_projects_by_tags, search_projects_semantic, search_projects_fulltext

# --- 1. Define State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    student_id: int
    profile_data: Dict[str, Any]
    final_output: str
    
    # RAG Workflow Data
    keywords: List[str]          # Extracted keywords
    interest_ids: List[int]      # Extracted interest tag IDs
    skill_ids: List[int]         # Extracted skill tag IDs
    interest_tags: List[Dict]    # Extracted interest tags (structured)
    skill_tags: List[Dict]       # Extracted skill tags (structured)
    
    # Parallel Track Results
    tag_candidates: List[Dict]
    semantic_candidates: List[Dict]
    keyword_candidates: List[Dict]
    
    # Final Ranked List
    ranked_projects: List[Dict]

# --- 2. Define Nodes ---

async def analyze_query_node(state: AgentState):
    """
    Step 1: Analyze user input to extract Keywords and Tags.
    """
    user_input = state['user_input']
    
    # Handle list input
    if isinstance(user_input, list):
        text_parts = []
        for block in user_input:
            if isinstance(block, dict) and block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif isinstance(block, str):
                text_parts.append(block)
        user_input = "\n".join(text_parts)
    elif not isinstance(user_input, str):
        user_input = str(user_input)
        
    # 1. Extract Keywords (LLM)
    keywords = await extract_keywords.ainvoke(user_input)
    
    # 2. Extract Tags (Vector Search in Tag KB)
    # We construct queries from user input + extracted keywords
    queries = [user_input]
    if isinstance(keywords, list):
        queries.extend(keywords)
    elif isinstance(keywords, str):
        queries.append(keywords)
        
    tag_result = await retrieve_tags.ainvoke({"queries": queries})
    
    return {
        "user_input": user_input,
        "keywords": keywords,
        "interest_ids": tag_result['interest_ids'],
        "skill_ids": tag_result['skill_ids'],
        "interest_tags": tag_result.get('interest_tags', []),
        "skill_tags": tag_result.get('skill_tags', [])
    }

async def track_tag_recall(state: AgentState):
    """Track 1: Tag-based (Precise)"""
    interest_ids = state.get('interest_ids', [])
    skill_ids = state.get('skill_ids', [])
    
    results = await search_projects_by_tags.ainvoke({
        "interest_ids": interest_ids, 
        "skill_ids": skill_ids
    })
    for p in results: p["source"] = "tag"
    return {"tag_candidates": results}

async def track_semantic_recall(state: AgentState):
    """Track 2: Semantic (Fuzzy)"""
    user_input = state['user_input']
    results = await search_projects_semantic.ainvoke({
        "query": user_input, 
        "k": 5
    })
    for p in results: p["source"] = "semantic"
    return {"semantic_candidates": results}

async def track_keyword_recall(state: AgentState):
    """Track 3: Keyword (Literal)"""
    keywords = state.get('keywords', [])
    results = await search_projects_fulltext.ainvoke({
        "keywords": keywords, 
        "k": 5
    })
    for p in results: p["source"] = "keyword"
    return {"keyword_candidates": results}

async def rerank_node(state: AgentState):
    """
    Step 3: Merge and Rerank.
    Includes content-based deduplication.
    """
    # Reranking is mostly CPU bound, but making it async ensures consistent graph execution
    # and allows yielding control if needed.
    candidates = {}
    
    def merge(source_list, source_name):
        if not source_list: return
        for p in source_list:
            pid = p['id']
            if pid not in candidates:
                candidates[pid] = p
            else:
                # Merge scores
                if source_name == "semantic":
                    candidates[pid]["semantic_score"] = p.get("score", 0)
                elif source_name == "keyword":
                    candidates[pid]["keyword_score"] = p.get("score", 0)
    
    merge(state.get("tag_candidates", []), "tag")
    merge(state.get("semantic_candidates", []), "semantic")
    merge(state.get("keyword_candidates", []), "keyword")
    
    candidates_list = []
    # Deduplication map: signature -> project_data
    # Signature can be (title + first 50 chars of description) to detect duplicates
    unique_map = {}
    
    for pid, data in candidates.items():
        tag_score = data.get("score", 0.0) if data.get("source") == "tag" else 0.0
        semantic_score = data.get("score", 0.0) if data.get("source") == "semantic" else data.get("semantic_score", 0.0)
        keyword_score = data.get("score", 0.0) if data.get("source") == "keyword" else data.get("keyword_score", 0.0)
        
        # Weighted Sum
        final_score = (tag_score * 0.4) + (semantic_score * 0.4) + (keyword_score * 0.2)
        data["final_score"] = final_score
        
        # Ensure title exists (Semantic might miss it if raw doc chunks are used directly without DB lookup)
        if "title" not in data: 
             continue
             
        # --- Content-based Deduplication ---
        # Generate a signature for the requirement content
        title = data.get('title', '').strip()
        desc = data.get('description', '').strip()
        # Create a simple signature: Title + first 100 chars of description (normalized)
        signature = f"{title.lower()}_{desc[:100].lower()}"
        
        if signature in unique_map:
            existing = unique_map[signature]
            # Keep the one with higher score, or just the first one encountered?
            # Let's keep the one with higher score
            if final_score > existing["final_score"]:
                unique_map[signature] = data
        else:
            unique_map[signature] = data
            
    # Convert unique map back to list
    candidates_list = list(unique_map.values())
        
    candidates_list.sort(key=lambda x: x["final_score"], reverse=True)
    return {"ranked_projects": candidates_list[:15]}


from langchain_core.runnables import RunnableConfig

async def reasoning_gen_node(state: AgentState, config: RunnableConfig):
    """
    Step 4: LLM Generation (Selection & Reasoning).
    """
    user_input = state['user_input']
    ranked_projects = state['ranked_projects']
    interest_tags = state.get('interest_tags', [])
    skill_tags = state.get('skill_tags', [])
    
    # Construct Context String for prompt
    tags_info = {
        "interest_tags": interest_tags,
        "skill_tags": skill_tags
    }
    tags_json = json.dumps(tags_info, ensure_ascii=False)
    
    # Serialize projects
    projects_json = json.dumps(ranked_projects, ensure_ascii=False)
    
    llm = Config.get_reasoning_llm()
    system_prompt = REASONING_GEN_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", REASONING_GEN_HUMAN_PROMPT)
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "user_input": user_input,
        "tags_info": tags_json,
        "projects": projects_json
    }, config=config)
    
    full_content = response.content
    
    # Return full content to frontend, let frontend handle parsing.
    # Frontend can extract <thinking> block for display and ignore JSON block.
    return {
        "messages": [AIMessage(content=full_content, name="reasoning")],
        "final_output": full_content
    }

async def reasoning_parse_node(state: AgentState):
    """Step 5: Parse JSON."""
    content = state['final_output']
    profile_data = {}
    try:
        json_str = ""
        # 1. Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 2. Fallback: Find JSON after </thinking>
            start_search_pos = 0
            thinking_end = content.find('</thinking>')
            if thinking_end != -1:
                start_search_pos = thinking_end + len('</thinking>')
            
            remaining_content = content[start_search_pos:]
            
            # Find first '{' and last '}'
            first_brace = remaining_content.find('{')
            last_brace = remaining_content.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = remaining_content[first_brace : last_brace + 1]
        
        if json_str:
            profile_data = json.loads(json_str)
    except:
        pass
        
    return {"profile_data": profile_data}

# --- 3. Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("analyze_query", analyze_query_node)
workflow.add_node("track_tag_recall", track_tag_recall)
workflow.add_node("track_semantic_recall", track_semantic_recall)
workflow.add_node("track_keyword_recall", track_keyword_recall)
workflow.add_node("rerank", rerank_node)
workflow.add_node("reasoning_gen", reasoning_gen_node)
workflow.add_node("reasoning_parse", reasoning_parse_node)

workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "track_tag_recall")
workflow.add_edge("analyze_query", "track_semantic_recall")
workflow.add_edge("analyze_query", "track_keyword_recall")
workflow.add_edge("track_tag_recall", "rerank")
workflow.add_edge("track_semantic_recall", "rerank")
workflow.add_edge("track_keyword_recall", "rerank")
workflow.add_edge("rerank", "reasoning_gen")
workflow.add_edge("reasoning_gen", "reasoning_parse")
workflow.add_edge("reasoning_parse", END)

app = workflow.compile()
