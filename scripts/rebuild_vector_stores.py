import csv
import logging
import os
import sys
from typing import Iterable, List

from pymilvus import Collection, connections, utility
from langchain_core.documents import Document


current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_root = os.path.dirname(current_dir)
if langchain_root not in sys.path:
    sys.path.append(langchain_root)

try:
    from core.django_setup import setup_django
    setup_django()
except ImportError:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(langchain_root, ".env"))

from core.config import Config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


VECTOR_COLLECTIONS = [
    "student_interests",
    "student_skills",
    "project_embeddings",
    "project_raw_docs",
]


def connect_milvus() -> None:
    connections.connect(alias="default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)


def drop_vector_collections() -> None:
    connect_milvus()
    existing = set(utility.list_collections())
    for name in VECTOR_COLLECTIONS:
        if name in existing:
            logger.info("Dropping collection: %s", name)
            utility.drop_collection(name)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def load_interest_docs() -> List[Document]:
    path = os.path.join(langchain_root, "tag_1.csv")
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            content = row.get("interest", "").strip()
            if not content:
                continue
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": int(row["id"]),
                        "tag_name": content,
                        "value": content,
                        "type": "interest",
                    },
                )
            )
    return docs


def load_skill_docs() -> List[Document]:
    path = os.path.join(langchain_root, "tag_2.csv")
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skill_name = row.get("skill", "").strip()
            if not skill_name:
                continue
            docs.append(
                Document(
                    page_content=skill_name,
                    metadata={
                        "id": int(row["id"]),
                        "tag_name": row.get("specialty", "").strip() or skill_name,
                        "post": skill_name,
                        "category": row.get("category", "").strip(),
                        "subcategory": row.get("subcategory", "").strip(),
                        "specialty": row.get("specialty", "").strip(),
                        "type": "skill",
                    },
                )
            )
    return docs


def fetch_projects() -> List[dict]:
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                title,
                brief,
                description,
                goal,
                expected_result,
                contact_person,
                contact_info,
                status
            FROM project_requirement
            WHERE status IN ('under_review', 'in_progress', 'completed')
            ORDER BY id
            """
        )
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


def build_project_summary(project: dict) -> str:
    parts = [
        f"Title: {project.get('title', '')}",
        f"Brief: {project.get('brief', '')}",
        f"Description: {project.get('description', '')}",
        f"Goal: {project.get('goal', '')}",
        f"Expected Result: {project.get('expected_result', '')}",
        f"Contact Person: {project.get('contact_person', '')}",
        f"Contact Info: {project.get('contact_info', '')}",
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def build_project_raw_text(project: dict) -> str:
    return build_project_summary(project)


def batched(items: List[Document], size: int) -> Iterable[List[Document]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def rebuild_tag_vectors() -> None:
    interest_docs = load_interest_docs()
    skill_docs = load_skill_docs()

    logger.info("Rebuilding student_interests with %s docs", len(interest_docs))
    if interest_docs:
        store = Config.get_milvus_store("student_interests")
        store.add_documents(interest_docs)

    logger.info("Rebuilding student_skills with %s docs", len(skill_docs))
    if skill_docs:
        store = Config.get_milvus_store("student_skills")
        store.add_documents(skill_docs)


def rebuild_project_vectors(batch_size: int = 50) -> None:
    projects = fetch_projects()
    logger.info("Rebuilding project vectors from %s projects", len(projects))

    summary_docs: List[Document] = []
    raw_docs: List[Document] = []

    for project in projects:
        project_id = project["id"]
        summary_text = build_project_summary(project)
        if summary_text:
            summary_docs.append(
                Document(
                    page_content=summary_text,
                    metadata={"project_id": project_id, "id": project_id},
                )
            )

        raw_text = build_project_raw_text(project)
        for chunk_index, chunk in enumerate(chunk_text(raw_text)):
            raw_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"project_id": project_id, "chunk_index": chunk_index},
                )
            )

    logger.info("Prepared %s summary docs", len(summary_docs))
    logger.info("Prepared %s raw chunk docs", len(raw_docs))

    summary_store = Config.get_milvus_store("project_embeddings")
    for batch in batched(summary_docs, batch_size):
        summary_store.add_documents(batch)

    raw_store = Config.get_milvus_store("project_raw_docs")
    for batch in batched(raw_docs, batch_size):
        raw_store.add_documents(batch)


def print_collection_status() -> None:
    connect_milvus()
    for name in VECTOR_COLLECTIONS:
        if not utility.has_collection(name):
            logger.info("%s: missing", name)
            continue
        collection = Collection(name)
        collection.flush()
        collection.load()
        logger.info("%s: num_entities=%s", name, collection.num_entities)


def main() -> None:
    logger.info(
        "Rebuilding Milvus vector stores with model=%s dim=%s",
        Config.get_text_embedding_target(),
        Config.get_text_embedding_dimension(),
    )
    drop_vector_collections()
    rebuild_tag_vectors()
    rebuild_project_vectors()
    print_collection_status()
    logger.info("Vector store rebuild complete.")


if __name__ == "__main__":
    main()
