import os
import sys
import logging
from typing import List, Dict

# Ensure langchain-v2.0 root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_root = os.path.dirname(current_dir)
if langchain_root not in sys.path:
    sys.path.append(langchain_root)

# Setup Django/Env
try:
    from core.django_setup import setup_django
    setup_django()
except ImportError:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(langchain_root, '.env'))

from core.config import Config
from langchain_core.documents import Document
from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def vectorize_projects():
    logger.info("Starting project vectorization...")
    
    # 1. Fetch Projects from MySQL
    conn = Config.get_db_connection()
    projects = []
    try:
        cursor = conn.cursor()
        # Fetch relevant projects
        query = """
            SELECT id, title, description, brief, status
            FROM project_requirement
            WHERE status IN ('under_review', 'in_progress', 'completed')
        """
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        for row in cursor.fetchall():
            projects.append(dict(zip(columns, row)))
        logger.info(f"Fetched {len(projects)} projects from database.")
    except Exception as e:
        logger.error(f"Database error: {e}")
        return
    finally:
        if conn:
            conn.close()

    if not projects:
        logger.warning("No projects to vectorize.")
        return

    # 2. Prepare Documents
    documents = []
    ids = []
    for p in projects:
        # Combine text for embedding
        text = f"{p['title']} {p.get('brief', '')} {p.get('description', '')}"
        
        # We need to store project_id in metadata/fields
        doc = Document(page_content=text, metadata={"project_id": p['id']})
        documents.append(doc)
        ids.append(p['id'])

    # 3. Setup Milvus Collection
    collection_name = "project_embeddings"
    dim = 1536 # v4
    
    # We use MilvusClient for easier management or Config.get_milvus_store
    # Config.get_milvus_store returns a LangChain VectorStore wrapper (Milvus)
    # But we might want to ensure the collection schema is correct first (drop old one)
    
    client = MilvusClient(uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
    
    if client.has_collection(collection_name):
        logger.info(f"Dropping existing collection '{collection_name}'...")
        client.drop_collection(collection_name)
        
    logger.info(f"Creating collection '{collection_name}' with dim={dim}...")
    # Define schema explicitly to match what we want
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="project_id", datatype=DataType.INT64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535) # Content
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    
    # 4. Vectorize and Insert
    logger.info("Generating embeddings and inserting...")
    
    # Use Config.get_embeddings() to get the embedding function
    embeddings_service = Config.get_embeddings()
    
    # Batch process
    batch_size = 50
    total = len(documents)
    
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_texts = [d.page_content for d in batch_docs]
        
        try:
            vectors = embeddings_service.embed_documents(batch_texts)
            
            data = []
            for j, vector in enumerate(vectors):
                data.append({
                    "id": batch_ids[j],
                    "vector": vector,
                    "project_id": batch_ids[j],
                    "text": batch_texts[j]
                })
                
            client.insert(collection_name=collection_name, data=data)
            logger.info(f"Processed {min(i+batch_size, total)}/{total}")
            
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")

    logger.info("Project vectorization complete.")

if __name__ == "__main__":
    vectorize_projects()
