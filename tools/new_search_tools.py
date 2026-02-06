import sys
import os
import json
from langchain_core.tools import tool
from core.config import Config

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@tool
def search_projects_by_tags(interest_ids: list[int], skill_ids: list[int]) -> list[dict]:
    """
    Track 1: Retrieve candidate projects based on Tag IDs (Precise Match).
    Returns list of dictionaries with scores.
    """
    conn = Config.get_db_connection()
    candidates = {} # {project_id: {data, score}}
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # 1. Search Interest Matches (Tag1)
        if interest_ids:
            format_strings = ','.join(['%s'] * len(interest_ids))
            # Assuming table is project_requirement_tag1 and fields are requirement_id, tag1_id
            query = f"""
                SELECT r.id, r.title, r.status, r.description 
                FROM project_requirement r
                JOIN project_requirement_tag1 rt ON r.id = rt.requirement_id
                WHERE rt.tag1_id IN ({format_strings})
                AND r.status IN ('under_review', 'in_progress', 'completed') 
            """
            
            cursor.execute(query, tuple(interest_ids))
            rows = cursor.fetchall()
            for row in rows:
                pid = row['id']
                if pid not in candidates:
                    candidates[pid] = {
                        "id": pid,
                        "title": row['title'],
                        "status": row['status'],
                        "description": row['description'],
                        "score": 0.0,
                        "match_type": "tag"
                    }
                candidates[pid]["score"] += 1.0 # 1 point per interest match

        # 2. Search Skill Matches (Tag2)
        if skill_ids:
            format_strings = ','.join(['%s'] * len(skill_ids))
            query = f"""
                SELECT r.id, r.title, r.status, r.description
                FROM project_requirement r
                JOIN project_requirement_tag2 rt ON r.id = rt.requirement_id
                WHERE rt.tag2_id IN ({format_strings})
                AND r.status IN ('under_review', 'in_progress', 'completed')
            """
            cursor.execute(query, tuple(skill_ids))
            rows = cursor.fetchall()
            for row in rows:
                pid = row['id']
                if pid not in candidates:
                    candidates[pid] = {
                        "id": pid,
                        "title": row['title'],
                        "status": row['status'],
                        "description": row['description'],
                        "score": 0.0,
                        "match_type": "tag"
                    }
                candidates[pid]["score"] += 1.0 # 1 point per skill match
                
    except Exception as e:
        print(f"Error in tag search: {e}")
    finally:
        if conn:
            conn.close()
            
    # Sort by score
    sorted_candidates = sorted(candidates.values(), key=lambda x: x['score'], reverse=True)
    return sorted_candidates[:20]

@tool
def search_projects_semantic(query: str) -> list[dict]:
    """
    Track 2: Retrieve candidate projects based on Semantic Similarity (Fuzzy Match).
    Uses 'project_embeddings' collection.
    """
    store = Config.get_milvus_store("project_embeddings")
    results = []
    milvus_matches = {} # {id: {score, doc}}
    
    try:
        # Search
        docs = store.similarity_search_with_score(query, k=20)
        
        project_ids = []
        for doc, score in docs:
            meta = doc.metadata
            pid = meta.get("project_id") or meta.get("id")
            if pid:
                project_ids.append(pid)
                milvus_matches[pid] = {"score": score, "content": doc.page_content}
        
        if not project_ids:
            return []
            
        # Fetch details from DB to fill title, status, etc.
        conn = Config.get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            format_strings = ','.join(['%s'] * len(project_ids))
            sql = f"""
                SELECT id, title, status, description 
                FROM project_requirement 
                WHERE id IN ({format_strings})
                AND status IN ('under_review', 'in_progress', 'completed')
            """
            cursor.execute(sql, tuple(project_ids))
            rows = cursor.fetchall()
            
            for row in rows:
                pid = row['id']
                if pid in milvus_matches:
                    match_info = milvus_matches[pid]
                    results.append({
                        "id": pid,
                        "title": row['title'],
                        "status": row['status'],
                        "description": row['description'], # Use clean DB description
                        "score": match_info['score'],
                        "match_type": "semantic"
                    })
        finally:
            if conn:
                conn.close()

    except Exception as e:
        print(f"Error in semantic search: {e}")
        
    return results

@tool
def search_projects_fulltext(keywords: list[str]) -> list[dict]:
    """
    Track 3: Retrieve candidate projects based on Fulltext Search (Keyword Match).
    Uses MySQL Fulltext Index.
    """
    if not keywords:
        return []
        
    conn = Config.get_db_connection()
    results = []
    try:
        cursor = conn.cursor(dictionary=True)
        # Prepare boolean query
        # e.g. "+keyword1 +keyword2" or just "keyword1 keyword2"
        search_query = " ".join(keywords) 
        
        # Using boolean mode for flexibility
        sql = f"""
            SELECT id, title, status, description, 
            MATCH(title, description) AGAINST (%s IN BOOLEAN MODE) as score
            FROM project_requirement
            WHERE MATCH(title, description) AGAINST (%s IN BOOLEAN MODE)
            AND status IN ('under_review', 'in_progress', 'completed')
            ORDER BY score DESC
            LIMIT 20
        """
        cursor.execute(sql, (search_query, search_query))
        rows = cursor.fetchall()
        
        # Fallback to LIKE if no results found with Fulltext (e.g. CJK issues)
        if not rows and keywords:
            # Construct LIKE query: title LIKE %k1% OR description LIKE %k1% ...
            conditions = []
            params = []
            for k in keywords:
                conditions.append("(title LIKE %s OR description LIKE %s)")
                params.extend([f"%{k}%", f"%{k}%"])
            
            where_clause = " OR ".join(conditions)
            sql_like = f"""
                SELECT id, title, status, description, 
                1.0 as score
                FROM project_requirement
                WHERE ({where_clause})
                AND status IN ('under_review', 'in_progress', 'completed')
                LIMIT 20
            """
            cursor.execute(sql_like, tuple(params))
            rows = cursor.fetchall()
            # Mark as LIKE match
            for row in rows:
                 row['match_type'] = 'like_fallback'

        for row in rows:
            results.append({
                "id": row['id'],
                "title": row['title'],
                "status": row['status'],
                "description": row['description'],
                "score": row['score'],
                "match_type": row.get('match_type', 'fulltext')
            })
            
    except Exception as e:
        print(f"Error in fulltext search: {e}")
    finally:
        if conn:
            conn.close()
            
    return results

@tool
def retrieve_project_chunks(project_ids: list[int], query: str) -> dict:
    """
    Retrieve detailed chunks for specific projects from 'project_raw_docs'.
    Returns {project_id_str: [chunk_texts]}.
    """
    if not project_ids:
        return {}
        
    store = Config.get_milvus_store("project_raw_docs")
    # Initialize with integer keys for processing
    temp_chunks = {pid: [] for pid in project_ids}
    
    try:
        # We can't easily filter by list of IDs in standard similarity_search without expr.
        # So we search broadly and filter, or use filter expression if supported.
        # LangChain Milvus supports 'expr'.
        
        expr = f"project_id in {project_ids}"
        # Search
        docs = store.similarity_search(query, k=50, expr=expr)
        
        for doc in docs:
            pid = doc.metadata.get("project_id")
            if pid in temp_chunks:
                temp_chunks[pid].append(doc.page_content)
                
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
    
    # Convert keys to strings for JSON serialization compatibility
    final_chunks = {str(k): v for k, v in temp_chunks.items()}
    return final_chunks

@tool
def retrieve_project_summary(project_ids: list[int], query: str) -> dict:
    """
    Retrieve summary/embedding chunks for specific projects from 'project_embeddings'.
    Returns {project_id_str: [chunk_texts]}.
    Fallback when raw docs are missing.
    """
    if not project_ids:
        return {}
        
    store = Config.get_milvus_store("project_embeddings")
    # Initialize with integer keys for processing
    temp_chunks = {pid: [] for pid in project_ids}
    
    try:
        expr = f"project_id in {project_ids}"
        # Search
        docs = store.similarity_search(query, k=10, expr=expr)
        
        for doc in docs:
            # Metadata might store ID as 'project_id' or 'id' depending on ingestion
            pid = doc.metadata.get("project_id") or doc.metadata.get("id")
            if pid in temp_chunks:
                temp_chunks[pid].append(doc.page_content)
                
    except Exception as e:
        print(f"Error retrieving summaries: {e}")
        
    # Convert keys to strings for JSON serialization compatibility
    final_chunks = {str(k): v for k, v in temp_chunks.items()}
    return final_chunks
