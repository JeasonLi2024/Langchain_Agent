import json
import sys
import os
from typing_extensions import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from core.config import Config
from core.prompts import PROJECT_QA_PROMPT
from tools.new_search_tools import retrieve_project_chunks, retrieve_project_summary

# --- Helper Function (Duplicated from main_agent.py to ensure isolation) ---
def get_last_message_text(messages: List[BaseMessage]) -> str:
    """Helper to extract text from the last message, handling list-based content."""
    if not messages:
        return ""
    last_msg = messages[-1]
    content = last_msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle list of content blocks (e.g. from multimodal models)
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return " ".join(text_parts)
    return str(content)

# --- 1. Define State ---
class QAState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    target_project_id: int
    user_info: dict  # Placeholder for user session info

# --- 2. Define Nodes ---

from langchain_core.runnables import RunnableConfig

def qa_node(state: QAState, config: RunnableConfig):
    """
    Standalone QA Node for specific projects.
    Retrieves chunks from Milvus (Raw Docs -> Embeddings fallback) and answers questions.
    Maintains conversation context.
    """
    target_id = state.get("target_project_id")
    messages = state["messages"]
    last_question = get_last_message_text(messages)
    llm = Config.get_utility_llm()
    
    # Check if target_id is present
    if not target_id:
        return {"messages": [AIMessage(content="错误：未提供目标项目ID。")]}

    # --- 1. Contextualize Question (History Management) ---
    chat_history = []
    # Take last 6 messages, excluding the very last one (current question)
    history_msgs = messages[:-1][-6:] 
    for msg in history_msgs:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Handle content extraction safely
        content = msg.content
        if isinstance(content, list):
             content = " ".join([b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"])
        chat_history.append(f"{role}: {content}")
    
    history_str = "\n".join(chat_history)
    
    # Generate standalone question for better retrieval
    standalone_question = last_question
    if history_str:
        condense_prompt = ChatPromptTemplate.from_template("""
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
            The standalone question should include necessary context (like what "it" refers to).
            Do NOT answer the question, just rewrite it.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            Standalone question:
        """)
        try:
            res = (condense_prompt | llm).invoke({
                "chat_history": history_str, 
                "question": last_question
            })
            standalone_question = res.content.strip()
        except:
            pass # Fallback to original
    
    # --- 2. Dual Retrieval ---
    # Strategy: Try Raw Docs -> If empty -> Try Embeddings (Summary)
    
    # A. Search Raw Docs
    context_chunks = retrieve_project_chunks.invoke({
        "project_ids": [target_id],
        "query": standalone_question
    })
    chunks_list = context_chunks.get(target_id, []) if context_chunks else []
    source = "Detailed Documents"
    
    # B. Fallback to Embeddings
    if not chunks_list:
        summary_chunks = retrieve_project_summary.invoke({
            "project_ids": [target_id],
            "query": standalone_question
        })
        chunks_list = summary_chunks.get(target_id, []) if summary_chunks else []
        source = "Project Summary"
    
    context_text = "\n\n".join(chunks_list)
    if not context_text:
        context_text = "No specific documents or summaries found for this project."
        
    # --- 3. Generate Answer ---
    qa_prompt = PROJECT_QA_PROMPT
    
    # Inject history into the question slot to provide context to the Answerer
    full_question_context = f"""
    Chat History:
    {history_str}
    
    Current Question: {last_question}
    """
    
    chain = ChatPromptTemplate.from_template(qa_prompt) | llm
    response = chain.invoke({
        "target_id": target_id,
        "context": f"Source: {source}\n\n{context_text}",
        "question": full_question_context
    }, config=config)
    
    return {"messages": [response]}

# --- 3. Build Graph ---
workflow = StateGraph(QAState)

workflow.add_node("qa_node", qa_node)

workflow.set_entry_point("qa_node")
workflow.add_edge("qa_node", END)

qa_app = workflow.compile()
