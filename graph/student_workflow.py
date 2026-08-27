import json
import re
from typing_extensions import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.prompts import (
    PROFILE_ANALYSIS_SYSTEM_PROMPT,
    PROFILE_ANALYSIS_HUMAN_PROMPT,
    PROFILE_SUMMARY_SYSTEM_PROMPT,
    PROFILE_SUMMARY_HUMAN_PROMPT,
    PROFILE_FOLLOWUP_SYSTEM_PROMPT,
    PROFILE_FOLLOWUP_HUMAN_PROMPT,
    RECOMMENDATION_REASONING_SYSTEM_PROMPT,
    RECOMMENDATION_REASONING_HUMAN_PROMPT,
)
from tools.search_tools import extract_keywords, retrieve_tags
from tools.new_search_tools import search_projects_by_tags, search_projects_semantic, search_projects_fulltext
from tools.db_tools import (
    load_student_context,
    update_student_core_fields,
    add_student_tags,
    upsert_student_profile_current,
    save_profile_evidence,
)


class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    user_id: int
    student_id: int
    thread_id: str
    profile_data: Dict[str, Any]
    final_output: str

    student_db_profile: Dict[str, Any]
    confirmed_interest_tags: List[Dict[str, Any]]
    confirmed_skill_tags: List[Dict[str, Any]]
    current_profile: Dict[str, Any]
    profile_draft: Dict[str, Any]
    student_pending_updates: Dict[str, Any]
    profile_missing_fields: List[str]
    profile_summary: str
    candidate_interest_tags: List[Dict[str, Any]]
    candidate_skill_tags: List[Dict[str, Any]]
    profile_evidence: List[Dict[str, Any]]
    profile_gate_decision: str

    keywords: List[str]
    interest_ids: List[int]
    skill_ids: List[int]
    interest_tags: List[Dict[str, Any]]
    skill_tags: List[Dict[str, Any]]

    tag_candidates: List[Dict[str, Any]]
    semantic_candidates: List[Dict[str, Any]]
    keyword_candidates: List[Dict[str, Any]]
    ranked_projects: List[Dict[str, Any]]


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p]).strip()
    if value is None:
        return ""
    return str(value).strip()


def _history_text(messages: List[BaseMessage], limit: int = 6) -> str:
    lines = []
    for msg in messages[-limit:]:
        role = getattr(msg, "type", "user")
        content = _normalize_text(getattr(msg, "content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _merge_unique_texts(*text_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for text_list in text_lists:
        for item in text_list or []:
            value = _normalize_text(item)
            if value and value not in seen:
                merged.append(value)
                seen.add(value)
    return merged


def _merge_tags(*tag_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for tag_list in tag_lists:
        for tag in tag_list or []:
            tag_id = tag.get("id")
            if tag_id is None:
                continue
            merged[tag_id] = {
                "id": tag_id,
                "name": tag.get("name") or tag.get("value") or tag.get("post"),
                **tag,
            }
    return list(merged.values())


def _extract_json_block(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    try:
        code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content, re.IGNORECASE)
        if code_match:
            return json.loads(code_match.group(1))
        brace_match = re.search(r"(\{[\s\S]*\})", content)
        if brace_match:
            return json.loads(brace_match.group(1))
    except Exception:
        return {}
    return {}


def _preview_student_profile(student_profile: Dict[str, Any], pending_updates: Dict[str, Any]) -> Dict[str, Any]:
    preview = dict(student_profile or {})
    for field in ("school_name", "major", "grade", "education_level"):
        if pending_updates.get(field):
            preview[field] = pending_updates[field]
    return preview


def _profile_ready(student_profile: Dict[str, Any], interest_tags: List[Dict[str, Any]], skill_tags: List[Dict[str, Any]], draft: Dict[str, Any], summary: str) -> bool:
    core_count = sum(
        1
        for field in ("school_name", "major", "grade", "education_level")
        if _normalize_text(student_profile.get(field))
    )
    has_tag_signal = bool(interest_tags or skill_tags or draft.get("interest_terms") or draft.get("skill_terms"))
    has_goal_signal = bool(_normalize_text(draft.get("target_direction")) or _normalize_text(draft.get("target_role")))
    has_experience_signal = bool(draft.get("experience_points"))
    return (core_count >= 2 and has_tag_signal and (has_goal_signal or has_experience_signal)) or bool(summary)


def _compute_missing_fields(student_profile: Dict[str, Any], confirmed_interest_tags: List[Dict[str, Any]], confirmed_skill_tags: List[Dict[str, Any]], draft: Dict[str, Any], summary: str) -> List[str]:
    missing = []
    if not _normalize_text(student_profile.get("school_name")):
        missing.append("school_name")
    if not _normalize_text(student_profile.get("major")):
        missing.append("major")
    if not _normalize_text(student_profile.get("grade")):
        missing.append("grade")
    if not _normalize_text(student_profile.get("education_level")):
        missing.append("education_level")

    if not confirmed_interest_tags and not draft.get("interest_terms"):
        missing.append("interest_direction")
    if not confirmed_skill_tags and not draft.get("skill_terms"):
        missing.append("skill_terms")
    if not _normalize_text(draft.get("target_direction")) and not _normalize_text(draft.get("target_role")):
        missing.append("target_direction")
    if not draft.get("experience_points") and not summary:
        missing.append("experience_points")

    priority = [
        "school_name",
        "major",
        "grade",
        "education_level",
        "interest_direction",
        "skill_terms",
        "target_direction",
        "experience_points",
    ]
    ordered = [field for field in priority if field in missing]
    return ordered[:3]


async def load_student_context_node(state: AgentState):
    result = json.loads(
        await load_student_context.ainvoke({
            "student_id": state.get("student_id"),
            "user_id": state.get("user_id"),
        })
    )

    if result.get("status") != "success":
        return {
            "student_db_profile": {},
            "confirmed_interest_tags": [],
            "confirmed_skill_tags": [],
            "current_profile": {},
            "profile_summary": "",
        }

    current_profile = result.get("current_profile", {}) or {}
    return {
        "student_id": result.get("student_id") or state.get("student_id"),
        "user_id": result.get("user_id") or state.get("user_id"),
        "student_db_profile": result.get("student_profile", {}) or {},
        "confirmed_interest_tags": result.get("confirmed_interest_tags", []) or [],
        "confirmed_skill_tags": result.get("confirmed_skill_tags", []) or [],
        "current_profile": current_profile,
        "profile_summary": state.get("profile_summary") or current_profile.get("profile_summary", ""),
    }


async def analyze_query_node(state: AgentState, config: RunnableConfig):
    user_input = _normalize_text(state.get("user_input"))
    llm = Config.get_utility_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILE_ANALYSIS_SYSTEM_PROMPT),
        ("human", PROFILE_ANALYSIS_HUMAN_PROMPT),
    ])

    response = await (prompt | llm).ainvoke({
        "user_input": user_input,
        "history_text": _history_text(state.get("messages", [])),
        "student_profile_json": json.dumps(state.get("student_db_profile", {}), ensure_ascii=False),
        "confirmed_interest_tags_json": json.dumps(state.get("confirmed_interest_tags", []), ensure_ascii=False),
        "confirmed_skill_tags_json": json.dumps(state.get("confirmed_skill_tags", []), ensure_ascii=False),
        "current_profile_summary": state.get("profile_summary", ""),
        "profile_draft_json": json.dumps(state.get("profile_draft", {}), ensure_ascii=False),
    }, config=config)

    analysis = _extract_json_block(_normalize_text(response.content))
    student_updates = analysis.get("student_updates", {}) or {}
    profile_updates = analysis.get("profile_updates", {}) or {}
    previous_draft = state.get("profile_draft", {}) or {}

    merged_draft = dict(previous_draft)
    for key in ("target_direction", "target_role"):
        value = _normalize_text(profile_updates.get(key))
        if value:
            merged_draft[key] = value

    merged_draft["interest_terms"] = _merge_unique_texts(previous_draft.get("interest_terms", []), profile_updates.get("interest_terms", []))
    merged_draft["skill_terms"] = _merge_unique_texts(previous_draft.get("skill_terms", []), profile_updates.get("skill_terms", []))
    merged_draft["experience_points"] = _merge_unique_texts(previous_draft.get("experience_points", []), profile_updates.get("experience_points", []))

    student_preview = _preview_student_profile(state.get("student_db_profile", {}), student_updates)
    current_summary = state.get("profile_summary", "")
    missing_fields = _compute_missing_fields(
        student_preview,
        state.get("confirmed_interest_tags", []),
        state.get("confirmed_skill_tags", []),
        merged_draft,
        current_summary,
    )

    evidence_text = _normalize_text(analysis.get("evidence_text")) or user_input
    evidence_item = {
        "evidence_type": "dialogue",
        "evidence_text": evidence_text,
        "field_mapping_json": json.dumps({
            "student_updates": {k: v for k, v in student_updates.items() if v},
            "profile_updates": merged_draft,
        }, ensure_ascii=False),
        "evidence_confidence": 0.85,
        "source": "agent_dialogue",
    }

    return {
        "user_input": user_input,
        "student_pending_updates": student_updates,
        "profile_draft": merged_draft,
        "profile_missing_fields": missing_fields,
        "profile_evidence": [evidence_item],
    }


async def persist_profile_updates_node(state: AgentState):
    student_id = state.get("student_id")
    if not student_id:
        return {}

    pending = state.get("student_pending_updates", {}) or {}
    if any(_normalize_text(pending.get(key)) for key in ("school_name", "major", "grade", "education_level")):
        await update_student_core_fields.ainvoke({
            "student_id": student_id,
            "school_name": pending.get("school_name"),
            "major": pending.get("major"),
            "grade": pending.get("grade"),
            "education_level": pending.get("education_level"),
        })

    for evidence in state.get("profile_evidence", []) or []:
        await save_profile_evidence.ainvoke({
            "student_id": student_id,
            "evidence_type": evidence.get("evidence_type", "dialogue"),
            "evidence_text": evidence.get("evidence_text", ""),
            "field_mapping_json": evidence.get("field_mapping_json", "{}"),
            "evidence_confidence": evidence.get("evidence_confidence", 0.8),
            "source": evidence.get("source", "agent"),
            "session_id": state.get("thread_id"),
        })

    refreshed_context = json.loads(
        await load_student_context.ainvoke({"student_id": student_id})
    )
    if refreshed_context.get("status") == "success":
        return {
            "student_db_profile": refreshed_context.get("student_profile", {}) or state.get("student_db_profile", {}),
            "confirmed_interest_tags": refreshed_context.get("confirmed_interest_tags", []) or state.get("confirmed_interest_tags", []),
            "confirmed_skill_tags": refreshed_context.get("confirmed_skill_tags", []) or state.get("confirmed_skill_tags", []),
        }
    return {}


async def build_candidate_tags_node(state: AgentState):
    user_input = state.get("user_input", "")
    draft = state.get("profile_draft", {}) or {}

    keywords = await extract_keywords.ainvoke(user_input)
    queries = [user_input]
    queries.extend(draft.get("interest_terms", []))
    queries.extend(draft.get("skill_terms", []))
    if draft.get("target_direction"):
        queries.append(draft["target_direction"])
    if state.get("profile_summary"):
        queries.append(state["profile_summary"])
    if isinstance(keywords, list):
        queries.extend(keywords)

    queries = [q for q in _merge_unique_texts(queries) if q]
    tag_result = await retrieve_tags.ainvoke({"queries": queries or [user_input]})

    merged_interest_tags = _merge_tags(
        state.get("confirmed_interest_tags", []),
        state.get("current_profile", {}).get("candidate_interest_tags", []),
        tag_result.get("interest_tags", []),
    )
    merged_skill_tags = _merge_tags(
        state.get("confirmed_skill_tags", []),
        state.get("current_profile", {}).get("candidate_skill_tags", []),
        tag_result.get("skill_tags", []),
    )

    confirmed_interest_ids = {tag.get("id") for tag in state.get("confirmed_interest_tags", [])}
    confirmed_skill_ids = {tag.get("id") for tag in state.get("confirmed_skill_tags", [])}
    new_interest_ids = [tag.get("id") for tag in merged_interest_tags if tag.get("id") not in confirmed_interest_ids][:3]
    new_skill_ids = [tag.get("id") for tag in merged_skill_tags if tag.get("id") not in confirmed_skill_ids][:5]
    if new_interest_ids or new_skill_ids:
        await add_student_tags.ainvoke({
            "student_id": state.get("student_id"),
            "interest_ids": new_interest_ids,
            "skill_ids": new_skill_ids,
        })

    refreshed_context = json.loads(
        await load_student_context.ainvoke({"student_id": state.get("student_id")})
    )
    confirmed_interest_tags = refreshed_context.get("confirmed_interest_tags", []) if refreshed_context.get("status") == "success" else state.get("confirmed_interest_tags", [])
    confirmed_skill_tags = refreshed_context.get("confirmed_skill_tags", []) if refreshed_context.get("status") == "success" else state.get("confirmed_skill_tags", [])

    merged_interest_tags = _merge_tags(confirmed_interest_tags, merged_interest_tags)
    merged_skill_tags = _merge_tags(confirmed_skill_tags, merged_skill_tags)

    return {
        "keywords": keywords if isinstance(keywords, list) else [keywords],
        "candidate_interest_tags": merged_interest_tags,
        "candidate_skill_tags": merged_skill_tags,
        "confirmed_interest_tags": confirmed_interest_tags,
        "confirmed_skill_tags": confirmed_skill_tags,
        "interest_ids": [tag.get("id") for tag in merged_interest_tags if tag.get("id") is not None],
        "skill_ids": [tag.get("id") for tag in merged_skill_tags if tag.get("id") is not None],
        "interest_tags": merged_interest_tags,
        "skill_tags": merged_skill_tags,
    }


async def build_profile_summary_node(state: AgentState, config: RunnableConfig):
    student_profile = state.get("student_db_profile", {}) or {}
    draft = state.get("profile_draft", {}) or {}
    ready = _profile_ready(
        student_profile,
        state.get("confirmed_interest_tags", []),
        state.get("confirmed_skill_tags", []),
        draft,
        state.get("profile_summary", ""),
    )
    if not ready:
        return {"profile_gate_decision": "need_profile_enrichment"}

    llm = Config.get_utility_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILE_SUMMARY_SYSTEM_PROMPT),
        ("human", PROFILE_SUMMARY_HUMAN_PROMPT),
    ])
    response = await (prompt | llm).ainvoke({
        "student_profile_json": json.dumps(student_profile, ensure_ascii=False),
        "confirmed_interest_tags_json": json.dumps(state.get("confirmed_interest_tags", []), ensure_ascii=False),
        "confirmed_skill_tags_json": json.dumps(state.get("confirmed_skill_tags", []), ensure_ascii=False),
        "profile_draft_json": json.dumps(draft, ensure_ascii=False),
    }, config=config)

    summary = _normalize_text(response.content)
    return {
        "profile_summary": summary[:300],
        "profile_gate_decision": "recommend_ready",
    }


async def persist_current_profile_node(state: AgentState):
    student_id = state.get("student_id")
    profile_summary = _normalize_text(state.get("profile_summary"))
    if not student_id or not profile_summary:
        return {}

    await upsert_student_profile_current.ainvoke({
        "student_id": student_id,
        "profile_summary": profile_summary[:300],
        "candidate_interest_tags_json": json.dumps(state.get("candidate_interest_tags", []), ensure_ascii=False),
        "candidate_skill_tags_json": json.dumps(state.get("candidate_skill_tags", []), ensure_ascii=False),
    })
    return {}


async def profile_gate_node(state: AgentState):
    student_profile = state.get("student_db_profile", {}) or {}
    draft = state.get("profile_draft", {}) or {}
    profile_summary = state.get("profile_summary", "")
    ready = _profile_ready(
        student_profile,
        state.get("confirmed_interest_tags", []),
        state.get("confirmed_skill_tags", []),
        draft,
        profile_summary,
    )
    missing_fields = _compute_missing_fields(
        student_profile,
        state.get("confirmed_interest_tags", []),
        state.get("confirmed_skill_tags", []),
        draft,
        profile_summary,
    )
    return {
        "profile_missing_fields": missing_fields,
        "profile_gate_decision": "recommend_ready" if ready else "need_profile_enrichment",
    }


async def ask_followup_node(state: AgentState, config: RunnableConfig):
    llm = Config.get_utility_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILE_FOLLOWUP_SYSTEM_PROMPT),
        ("human", PROFILE_FOLLOWUP_HUMAN_PROMPT),
    ])
    response = await (prompt | llm).ainvoke({
        "student_profile_json": json.dumps(state.get("student_db_profile", {}), ensure_ascii=False),
        "confirmed_interest_tags_json": json.dumps(state.get("confirmed_interest_tags", []), ensure_ascii=False),
        "confirmed_skill_tags_json": json.dumps(state.get("confirmed_skill_tags", []), ensure_ascii=False),
        "current_profile_summary": state.get("profile_summary", ""),
        "profile_draft_json": json.dumps(state.get("profile_draft", {}), ensure_ascii=False),
        "missing_fields_json": json.dumps(state.get("profile_missing_fields", []), ensure_ascii=False),
    }, config=config)
    content = _normalize_text(response.content)
    return {
        "messages": [AIMessage(content=content)],
        "final_output": content,
        "profile_data": {},
    }


def _route_after_gate(state: AgentState):
    return state.get("profile_gate_decision", "need_profile_enrichment")


async def recall_ready_node(state: AgentState):
    return {}


async def track_tag_recall(state: AgentState):
    results = await search_projects_by_tags.ainvoke({
        "interest_ids": state.get("interest_ids", []),
        "skill_ids": state.get("skill_ids", []),
        "k": 20,
    })
    for project in results:
        project["source"] = "tag"
    return {"tag_candidates": results}


async def track_semantic_recall(state: AgentState):
    profile_summary = _normalize_text(state.get("profile_summary"))
    query = "；".join([part for part in [profile_summary, state.get("user_input", "")] if part]).strip()
    results = await search_projects_semantic.ainvoke({
        "query": query or state.get("user_input", ""),
        "k": 20,
    })
    for project in results:
        project["source"] = "semantic"
    return {"semantic_candidates": results}


async def track_keyword_recall(state: AgentState):
    student_profile = state.get("student_db_profile", {}) or {}
    draft = state.get("profile_draft", {}) or {}
    keyword_terms = _merge_unique_texts(
        state.get("keywords", []),
        [student_profile.get("school_name"), student_profile.get("major"), draft.get("target_direction"), draft.get("target_role")],
        draft.get("interest_terms", []),
        draft.get("skill_terms", []),
        draft.get("experience_points", []),
    )
    results = await search_projects_fulltext.ainvoke({
        "keywords": keyword_terms,
        "k": 20,
    })
    for project in results:
        project["source"] = "keyword"
    return {"keyword_candidates": results}


async def rerank_node(state: AgentState):
    candidates: Dict[int, Dict[str, Any]] = {}

    def merge(source_list: List[Dict[str, Any]], source_name: str):
        for project in source_list or []:
            pid = project["id"]
            existing = candidates.setdefault(pid, dict(project))
            if source_name == "semantic":
                existing["semantic_score"] = project.get("score", 0)
            elif source_name == "keyword":
                existing["keyword_score"] = project.get("score", 0)
            elif source_name == "tag":
                existing["tag_score"] = project.get("score", 0)

    merge(state.get("tag_candidates", []), "tag")
    merge(state.get("semantic_candidates", []), "semantic")
    merge(state.get("keyword_candidates", []), "keyword")

    ranked: List[Dict[str, Any]] = []
    summary = _normalize_text(state.get("profile_summary")).lower()
    draft = state.get("profile_draft", {}) or {}
    target_direction = _normalize_text(draft.get("target_direction")).lower()
    skill_names = [(_normalize_text(tag.get("name"))).lower() for tag in state.get("candidate_skill_tags", [])]

    for _, project in candidates.items():
        title = _normalize_text(project.get("title"))
        description = _normalize_text(project.get("description"))
        tag_score = project.get("tag_score", project.get("score", 0.0) if project.get("source") == "tag" else 0.0)
        semantic_score = project.get("semantic_score", project.get("score", 0.0) if project.get("source") == "semantic" else 0.0)
        keyword_score = project.get("keyword_score", project.get("score", 0.0) if project.get("source") == "keyword" else 0.0)

        boost = 0.0
        project_text = f"{title}\n{description}".lower()
        if target_direction and target_direction in project_text:
            boost += 0.15
        if summary:
            for snippet in summary.split("，")[:4]:
                snippet = snippet.strip()
                if len(snippet) >= 4 and snippet.lower() in project_text:
                    boost += 0.03
        skill_hits = sum(1 for name in skill_names[:6] if name and name in project_text)
        boost += min(skill_hits * 0.05, 0.2)

        project["final_score"] = (tag_score * 0.4) + (semantic_score * 0.4) + (keyword_score * 0.2) + boost
        ranked.append(project)

    ranked.sort(key=lambda item: item.get("final_score", 0), reverse=True)
    return {"ranked_projects": ranked[:15]}


async def reasoning_gen_node(state: AgentState, config: RunnableConfig):
    llm = Config.get_reasoning_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RECOMMENDATION_REASONING_SYSTEM_PROMPT),
        ("human", RECOMMENDATION_REASONING_HUMAN_PROMPT),
    ])
    tags_info = {
        "confirmed_interest_tags": state.get("confirmed_interest_tags", []),
        "confirmed_skill_tags": state.get("confirmed_skill_tags", []),
        "candidate_interest_tags": state.get("candidate_interest_tags", []),
        "candidate_skill_tags": state.get("candidate_skill_tags", []),
    }
    response = await (prompt | llm).ainvoke({
        "user_input": state.get("user_input", ""),
        "profile_summary": state.get("profile_summary", ""),
        "tags_info": json.dumps(tags_info, ensure_ascii=False),
        "projects": json.dumps(state.get("ranked_projects", []), ensure_ascii=False),
    }, config=config)
    full_content = _normalize_text(response.content)
    return {
        "messages": [AIMessage(content=full_content, name="reasoning")],
        "final_output": full_content,
    }


async def reasoning_parse_node(state: AgentState):
    profile_data = _extract_json_block(state.get("final_output", ""))
    return {"profile_data": profile_data}


workflow = StateGraph(AgentState)

workflow.add_node("load_student_context", load_student_context_node)
workflow.add_node("analyze_query", analyze_query_node)
workflow.add_node("persist_profile_updates", persist_profile_updates_node)
workflow.add_node("build_candidate_tags", build_candidate_tags_node)
workflow.add_node("build_profile_summary", build_profile_summary_node)
workflow.add_node("persist_current_profile", persist_current_profile_node)
workflow.add_node("profile_gate", profile_gate_node)
workflow.add_node("ask_followup", ask_followup_node)
workflow.add_node("recall_ready", recall_ready_node)
workflow.add_node("track_tag_recall", track_tag_recall)
workflow.add_node("track_semantic_recall", track_semantic_recall)
workflow.add_node("track_keyword_recall", track_keyword_recall)
workflow.add_node("rerank", rerank_node)
workflow.add_node("reasoning_gen", reasoning_gen_node)
workflow.add_node("reasoning_parse", reasoning_parse_node)

workflow.set_entry_point("load_student_context")
workflow.add_edge("load_student_context", "analyze_query")
workflow.add_edge("analyze_query", "persist_profile_updates")
workflow.add_edge("persist_profile_updates", "build_candidate_tags")
workflow.add_edge("build_candidate_tags", "build_profile_summary")
workflow.add_edge("build_profile_summary", "persist_current_profile")
workflow.add_edge("persist_current_profile", "profile_gate")
workflow.add_conditional_edges(
    "profile_gate",
    _route_after_gate,
    {
        "need_profile_enrichment": "ask_followup",
        "recommend_ready": "recall_ready",
    },
)
workflow.add_edge("ask_followup", END)
workflow.add_edge("recall_ready", "track_tag_recall")
workflow.add_edge("recall_ready", "track_semantic_recall")
workflow.add_edge("recall_ready", "track_keyword_recall")
workflow.add_edge("track_tag_recall", "rerank")
workflow.add_edge("track_semantic_recall", "rerank")
workflow.add_edge("track_keyword_recall", "rerank")
workflow.add_edge("rerank", "reasoning_gen")
workflow.add_edge("reasoning_gen", "reasoning_parse")
workflow.add_edge("reasoning_parse", END)

app = workflow.compile()
