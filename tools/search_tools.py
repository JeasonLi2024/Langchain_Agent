import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from core.config import Config

@tool
def extract_keywords(user_input: str) -> list[str]:
    """Extract 3-5 distinct technical keywords from user description."""
    llm = Config.get_utility_llm()
    prompt = f"""
    Extract 3-5 distinct technical keywords or phrases from the user's description.
    Return ONLY the keywords separated by commas.
    User description: {user_input}
    """
    try:
        response = llm.invoke(prompt)
        keywords = [k.strip() for k in response.content.split(',') if k.strip()]
        return keywords
    except Exception as e:
        return [user_input]

@tool
def retrieve_tags(queries: list[str]) -> dict:
    """Retrieve relevant interest and skill tags from Milvus based on queries."""
    interest_store = Config.get_milvus_store("student_interests")
    skill_store = Config.get_milvus_store("student_skills")
    
    interest_results = {}
    skill_results = {}
    
    for q in queries:
        try:
            # Interest Search
            docs_int = interest_store.similarity_search_with_score(q, k=5)
            for doc, score in docs_int:
                doc_id = doc.metadata.get('id')
                if doc_id not in interest_results:
                    interest_results[doc_id] = (doc, score)
            
            # Skill Search
            docs_skill = skill_store.similarity_search_with_score(q, k=5)
            for doc, score in docs_skill:
                doc_id = doc.metadata.get('id')
                if doc_id not in skill_results:
                    skill_results[doc_id] = (doc, score)
        except Exception as e:
            print(f"Error in retrieval for '{q}': {e}")

    # Format Output
    context_str = "Matched Interest Tags:\n"
    cand_int_ids = []
    interest_tags = []

    # Sort by score descending
    sorted_interest = sorted(interest_results.values(), key=lambda x: x[1], reverse=True)[:6]
    
    for doc, score in sorted_interest:
        meta = doc.metadata
        tid = meta.get('id')
        cand_int_ids.append(tid)
        # Use 'value' field for interest tags (Tag1) which contains full "Domain-SubDomain" format
        tag_name = meta.get('value') or meta.get('tag_name') 
        context_str += f"[ID: {tid}] Type: {meta.get('type')} - Name: {tag_name} (Score: {score:.4f})\n"
        interest_tags.append({
            "id": tid,
            "name": tag_name,
            "Similarity Score": float(score)
        })

    context_str += "\nMatched Skill Tags:\n"
    cand_skill_ids = []
    skill_tags = []

    # Sort by score descending
    sorted_skill = sorted(skill_results.values(), key=lambda x: x[1], reverse=True)[:6]

    for doc, score in sorted_skill:
        meta = doc.metadata
        tid = meta.get('id')
        cand_skill_ids.append(tid)
        # Use 'post' field for skill tags (Tag2) which contains full "Category-SubCategory-Skill" format
        tag_name = meta.get('post') or meta.get('tag_name')
        context_str += f"[ID: {tid}] Type: {meta.get('type')} - Name: {tag_name} (Score: {score:.4f})\n"
        skill_tags.append({
            "id": tid,
            "name": tag_name,
            "Similarity Score": float(score)
        })
        
    return {
        "context_str": context_str,
        "interest_ids": cand_int_ids,
        "skill_ids": cand_skill_ids,
        "interest_tags": interest_tags,
        "skill_tags": skill_tags
    }

@tool
def retrieve_project_details(query: str) -> str:
    """
    Retrieve project details from Milvus (project_raw_docs) based on query.
    IMPORTANT: Performs strict status filtering to exclude draft/private projects.
    """
    try:
        store = Config.get_milvus_store("project_raw_docs")
        # Search more candidates to allow for filtering
        docs = store.similarity_search_with_score(query, k=20)
        
        if not docs:
            return "No relevant project details found."
            
        # Collect Project IDs
        project_ids = set()
        for doc, score in docs:
            pid = doc.metadata.get('project_id')
            if pid:
                project_ids.add(pid)
        
        if not project_ids:
            return "No relevant project details found."
            
        # Filter by Status via MySQL
        conn = Config.get_db_connection()
        valid_ids = set()
        try:
            cursor = conn.cursor()
            format_strings = ','.join(['%s'] * len(project_ids))
            sql = f"""
            SELECT id FROM project_requirement 
            WHERE id IN ({format_strings}) 
            AND status IN ('in_progress', 'completed', 'paused')
            """
            cursor.execute(sql, tuple(project_ids))
            for r in cursor.fetchall():
                valid_ids.add(r[0])
        finally:
            conn.close()
            
        # Filter docs
        results = []
        for doc, score in docs:
            pid = doc.metadata.get('project_id')
            if pid in valid_ids:
                content = doc.page_content
                # Truncate if too long
                results.append(f"[Project ID: {pid}] {content[:300]}...")
                
        if not results:
            return "No relevant public projects found."
            
        return "\n\n".join(results[:5])
        
    except Exception as e:
        return f"Error retrieving projects: {str(e)}"
