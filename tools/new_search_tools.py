import sys
import os
import json
import pymysql
from langchain_core.tools import tool
from core.config import Config
from core.embedding_service import EmbeddingService
from pymilvus import MilvusClient

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@tool
def search_projects_by_tags(interest_ids: list[int], skill_ids: list[int]) -> list[dict]:
    """
    Track 1: Retrieve candidate projects based on Tag IDs (Precise Match).
    Queries each tag individually and counts project occurrence frequency.
    Returns top 5 projects with highest occurrence frequency.
    """
    conn = Config.get_db_connection()
    project_counts = {} # {project_id: count}
    project_details = {} # {project_id: data}
    
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 1. Search Interest Matches (Tag1) - Query individually
        for tid in interest_ids:
            query = """
                SELECT r.id, r.title, r.status, r.description 
                FROM project_requirement r
                JOIN project_requirement_tag1 rt ON r.id = rt.requirement_id
                WHERE rt.tag1_id = %s
                AND r.status IN ('in_progress', 'completed', 'paused') 
            """
            cursor.execute(query, (tid,))
            rows = cursor.fetchall()
            for row in rows:
                pid = row['id']
                project_counts[pid] = project_counts.get(pid, 0) + 1
                if pid not in project_details:
                    project_details[pid] = {
                        "id": pid,
                        "title": row['title'],
                        "status": row['status'],
                        "description": row['description'],
                        "match_type": "tag"
                    }

        # 2. Search Skill Matches (Tag2) - Query individually
        for tid in skill_ids:
            query = """
                SELECT r.id, r.title, r.status, r.description
                FROM project_requirement r
                JOIN project_requirement_tag2 rt ON r.id = rt.requirement_id
                WHERE rt.tag2_id = %s
                AND r.status IN ('in_progress', 'completed', 'paused')
            """
            cursor.execute(query, (tid,))
            rows = cursor.fetchall()
            for row in rows:
                pid = row['id']
                project_counts[pid] = project_counts.get(pid, 0) + 1
                if pid not in project_details:
                    project_details[pid] = {
                        "id": pid,
                        "title": row['title'],
                        "status": row['status'],
                        "description": row['description'],
                        "match_type": "tag"
                    }
                
    except Exception as e:
        print(f"Error in tag search: {e}")
    finally:
        if conn:
            conn.close()
            
    # Sort by count (frequency)
    sorted_pids = sorted(project_counts.keys(), key=lambda pid: project_counts[pid], reverse=True)
    top_pids = sorted_pids[:5]
    
    results = []
    for pid in top_pids:
        data = project_details[pid]
        data['score'] = float(project_counts[pid]) # Use count as score
        results.append(data)
        
    return results

@tool
def search_projects_semantic(query: str, k: int = 5) -> list[dict]:
    """
    Track 2: Retrieve candidate projects based on Semantic Similarity (Fuzzy Match).
    Uses 'project_embeddings' collection.
    Returns top 5 projects.
    """
    results = []
    try:
        # Generate embedding for query
        # EmbeddingService handles the dimension logic
        embeddings = EmbeddingService.get_embeddings([query])
        if not embeddings:
            return []
        query_vector = embeddings[0][1] if isinstance(embeddings[0], tuple) else embeddings[0]
        
        # Connect to Milvus
        client = MilvusClient(uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
        
        search_res = client.search(
            collection_name="project_embeddings",
            data=[query_vector],
            limit=k,
            output_fields=["project_id"] # Milvus only stores minimal fields
        )
        
        project_ids = []
        scores_map = {}
        
        for hits in search_res:
            for hit in hits:
                entity = hit.get("entity", {})
                pid = entity.get("project_id")
                if pid:
                    project_ids.append(pid)
                    scores_map[pid] = hit.get("distance")

        # Fetch metadata from MySQL
        if project_ids:
            conn = Config.get_db_connection()
            try:
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                format_strings = ','.join(['%s'] * len(project_ids))
                query = f"""
                    SELECT id, title, status, description 
                    FROM project_requirement 
                    WHERE id IN ({format_strings})
                    AND status IN ('in_progress', 'completed', 'paused')
                """
                cursor.execute(query, tuple(project_ids))
                rows = cursor.fetchall()
                
                for row in rows:
                    pid = row['id']
                    results.append({
                        "id": pid,
                        "title": row['title'],
                        "status": row['status'],
                        "description": row['description'],
                        "score": scores_map.get(pid, 0.0),
                        "match_type": "semantic"
                    })
            finally:
                if conn:
                    conn.close()
                
    except Exception as e:
        print(f"Error in semantic search: {e}")
        
    return results

@tool
def search_projects_fulltext(keywords: list[str], k: int = 5) -> list[dict]:
    """
    Track 3: Retrieve candidate projects based on Fulltext Search (Keyword Match).
    Counts how many UNIQUE keyword types are matched in title/brief/description.
    Returns top 5 projects with highest unique keyword match count.
    """
    if not keywords:
        return []
        
    conn = Config.get_db_connection()
    results = []
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # We need to check presence of each keyword.
        # Construct a query that returns the text fields for projects matching ANY keyword.
        # Then we process in Python to count unique types.
        
        search_query = " ".join(keywords) 
        
        # 1. Broad Recall (OR logic)
        # Using FULLTEXT index for better performance
        sql = """
            SELECT id, title, status, description, brief
            FROM project_requirement
            WHERE MATCH(title, brief, description) AGAINST (%s IN BOOLEAN MODE)
            AND status IN ('in_progress', 'completed', 'paused')
            LIMIT 100
        """
        cursor.execute(sql, (search_query,))
        rows = cursor.fetchall()
        
        scored_projects = []
        for row in rows:
            # 2. Calculate Unique Keyword Type Count
            unique_hits = 0
            text_content = (f"{row.get('title', '')} {row.get('brief', '')} {row.get('description', '')}").lower()
            
            for kw in keywords:
                if kw.lower() in text_content:
                    unique_hits += 1
            
            if unique_hits > 0:
                row['score'] = float(unique_hits)
                row['match_type'] = "fulltext"
                scored_projects.append(row)
                
        # 3. Sort and Limit
        scored_projects.sort(key=lambda x: x['score'], reverse=True)
        results = scored_projects[:k]
            
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
    Returns {project_id: [chunk_texts]}.
    """
    if not project_ids:
        return {}
        
    store = Config.get_milvus_store("project_raw_docs")
    project_chunks = {pid: [] for pid in project_ids}
    
    try:
        # We can't easily filter by list of IDs in standard similarity_search without expr.
        # So we search broadly and filter, or use filter expression if supported.
        # LangChain Milvus supports 'expr'.
        
        expr = f"project_id in {project_ids}"
        # Search
        docs = store.similarity_search(query, k=50, expr=expr)
        
        for doc in docs:
            pid = doc.metadata.get("project_id")
            if pid in project_chunks:
                project_chunks[pid].append(doc.page_content)
                
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        
    return project_chunks
