import os
import sys
import csv
import logging
from typing import List, Dict

# Ensure langchain-v2.0 root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_root = os.path.dirname(current_dir)
if langchain_root not in sys.path:
    sys.path.append(langchain_root)

# Prioritize Local Env
from core.config import Config

# Setup Django (Required for pymilvus connection via Config if needed, or just to load .env)
# Using core.django_setup to robustly load environment
try:
    from core.django_setup import setup_django
    setup_django()
except ImportError:
    # Fallback if running directly in a simplified env
    pass

from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv_tags(file_path: str, tag_type: str) -> List[Document]:
    """
    Load tags from CSV and convert to LangChain Documents.
    
    Args:
        file_path: Path to CSV file
        tag_type: 'tag1' (Interest) or 'tag2' (Skill)
    """
    documents = []
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tag_id = int(row['id'])
                
                if tag_type == 'tag1':
                    # Interest Tags: id, interest
                    content = row.get('interest', '').strip()
                    metadata = {
                        "id": tag_id,
                        "tag_name": content, # Standardize for retrieval
                        "value": content,    # Original field name
                        "type": "interest"
                    }
                else:
                    # Skill Tags: id, skill, category, subcategory, specialty
                    # We want to embed the full context string for better semantic matching
                    # e.g. "互联网 后端开发 java"
                    skill_name = row.get('skill', '').strip()
                    category = row.get('category', '').strip()
                    subcategory = row.get('subcategory', '').strip()
                    specialty = row.get('specialty', '').strip()
                    
                    # Construct rich embedding content
                    # Format: "Category Subcategory Specialty"
                    content = f"{skill_name}" 
                    
                    metadata = {
                        "id": tag_id,
                        "tag_name": specialty or skill_name,
                        "post": skill_name, # Original full path
                        "category": category,
                        "subcategory": subcategory,
                        "specialty": specialty,
                        "type": "skill"
                    }
                
                if content:
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipping row {row}: {e}")
                
    return documents

def vectorize_tags():
    """Main function to vectorize both tag files."""
    
    # 1. Define Paths
    tag1_path = os.path.join(langchain_root, "tag_1.csv")
    tag2_path = os.path.join(langchain_root, "tag_2.csv")
    
    # 2. Load Documents
    logger.info("Loading Interest Tags (Tag 1)...")
    docs_tag1 = load_csv_tags(tag1_path, 'tag1')
    logger.info(f"Loaded {len(docs_tag1)} interest tags.")
    
    logger.info("Loading Skill Tags (Tag 2)...")
    docs_tag2 = load_csv_tags(tag2_path, 'tag2')
    logger.info(f"Loaded {len(docs_tag2)} skill tags.")
    
    if not docs_tag1 and not docs_tag2:
        logger.error("No tags loaded. Exiting.")
        return

    # 3. Initialize Vector Stores
    # Using Config to get pre-configured Milvus instance with DashScope Embeddings
    
    # Collection names must match what retrieval_tool.py expects:
    # "student_interests" for Tag 1
    # "student_skills" for Tag 2
    
    if docs_tag1:
        logger.info("Vectorizing Interest Tags to collection 'student_interests'...")
        try:
            store_tag1 = Config.get_milvus_store("student_interests")
            # Milvus.from_documents automatically handles collection creation and insertion
            store_tag1.add_documents(docs_tag1)
            logger.info("Successfully vectorized Interest Tags.")
        except Exception as e:
            logger.error(f"Failed to vectorize Interest Tags: {e}")

    if docs_tag2:
        logger.info("Vectorizing Skill Tags to collection 'student_skills'...")
        try:
            store_tag2 = Config.get_milvus_store("student_skills")
            store_tag2.add_documents(docs_tag2)
            logger.info("Successfully vectorized Skill Tags.")
        except Exception as e:
            logger.error(f"Failed to vectorize Skill Tags: {e}")

if __name__ == "__main__":
    vectorize_tags()
