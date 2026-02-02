import sys
import os
import json
import re
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from core.config import Config
from core.prompts import TAG_RECOMMENDATION_SYSTEM_PROMPT, TAG_RECOMMENDATION_HUMAN_PROMPT
from tools.search_tools import extract_keywords, retrieve_tags

# Removed @tool decorator to use as a regular function in graph node
def recommend_tags_logic(description: str, research_direction: str, skill: str) -> str:
    """
    Recommend 3 interest tags and 5 skill tags based on project requirement details.
    Returns a string containing the thinking process and the final JSON result.
    """
    
    # 1. Construct query context
    query_text = f"{description} {research_direction} {skill}"
    
    # 2. Extract Keywords
    keywords = extract_keywords.invoke(query_text)
    queries = [query_text] + keywords
    
    # 3. Retrieve Candidate Tags
    retrieval_res = retrieve_tags.invoke({"queries": queries})
    context_str = retrieval_res['context_str']
    
    # 4. LLM Reasoning
    llm = Config.get_reasoning_llm()
    system_prompt = TAG_RECOMMENDATION_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", TAG_RECOMMENDATION_HUMAN_PROMPT)
    ])
    
    chain = prompt | llm
    
    # We use invoke here. For streaming, the calling node in LangGraph 
    # should be part of a stream capable graph. 
    # Since we want to stream tokens to the frontend, this node needs to be connected to the graph.
    response = chain.invoke({
        "query_text": query_text,
        "context": context_str
    })
    
    return response.content
