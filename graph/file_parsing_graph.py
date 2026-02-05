from typing import TypedDict, List, Dict, Any, Annotated
import logging
import os
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import Config # Ensure env vars are loaded
from core.embedding_service import get_or_create_collection, generate_embedding, COLLECTION_RAW_DOCS, EmbeddingService

logger = logging.getLogger(__name__)

# --- State Definition ---
class FileParsingState(TypedDict):
    file_path: str
    file_name: str
    draft_id: int  # Optional: if draft already exists
    
    # Outputs
    summary: str   # Summary for Agent
    chunks: List[str] # All parsed chunks
    chunk_embeddings: List[List[float]] # Embeddings for all chunks (aligned with chunks)
    filtered_chunks: List[str] # Top-N filtered chunks
    extracted_data: Dict[str, Any] # Structured data from LLM
    success: bool
    error: str

# --- Nodes ---

async def loader_node(state: FileParsingState):
    """Load file content based on extension."""
    file_path = state['file_path']
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}
        
    try:
        # File IO is blocking, but okay for local files.
        # Ideally wrap in run_in_executor if heavy.
        if file_path.lower().endswith('.pdf'):
            # Using PyPDFLoader for PDF which can extract text from tables if they are text-based.
            # If tables are images, OCR would be needed (e.g., Unstructured with OCR), 
            # but standard PyPDFLoader is text-only.
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            # UnstructuredWordDocumentLoader handles tables in DOCX well, converting them to text.
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.lower().endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            # Fallback for text files
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path)
            
        docs = loader.load()
        full_text = "\n\n".join([d.page_content for d in docs])
        return {"chunks": [full_text], "summary": full_text[:2000]} # Preliminary summary
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return {"success": False, "error": str(e)}

# EmbeddingService is already imported from core.embedding_service

async def cleaner_node(state: FileParsingState):
    """Clean and split text."""
    if not state.get('success', True):
        return {}
        
    raw_text = state['chunks'][0]
    
    # 1. Cleaning
    # Remove excessive newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    
    # 2. Splitting (Use project/services.py logic for consistency)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "!", ".", " "]
    )
    chunks = splitter.split_text(cleaned_text)
    
    # 3. Rule-based filtering
    valid_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) < 10: # Consistency with backend
            continue
        valid_chunks.append(chunk)
    
    return {"chunks": valid_chunks}

async def ranking_node(state: FileParsingState):
    """Rank and filter chunks using vector similarity."""
    if not state.get('success', True):
        return {}
        
    chunks = state['chunks']
    if not chunks:
        return {"filtered_chunks": []}
        
    # Query for target information
    query = "项目标题 项目简介 详细描述 研究方向 技术栈 完成时间 预算 资金支持"
    
    try:
        # Embed query
        query_vec = generate_embedding(query) # This is wrapper, let's keep it sync or make async wrapper. 
        # But generate_embedding is sync wrapper. Let's use EmbeddingService.aget_single_embedding if possible or just accept one sync call.
        # Ideally: query_vec = await EmbeddingService.aget_single_embedding(query) (I didn't add aget_single_embedding yet)
        # I'll just use aget_embeddings([query])[0]
        query_vecs = await EmbeddingService.aget_embeddings([query])
        query_vec = query_vecs[0] if query_vecs else None

        if not query_vec:
            logger.error("Failed to generate query embedding")
            return {"filtered_chunks": chunks[:10]}
            
        # Embed chunks (Use Async Batch API)
        chunk_vecs = await EmbeddingService.aget_embeddings(chunks)
        
        # Filter invalid embeddings
        valid_indices = [i for i, v in enumerate(chunk_vecs) if v and len(v) > 0]
        valid_chunk_vecs = [chunk_vecs[i] for i in valid_indices]
        valid_chunks = [chunks[i] for i in valid_indices]
        
        if not valid_chunk_vecs:
            return {"filtered_chunks": chunks[:10], "chunk_embeddings": []}
            
        # Calculate Cosine Similarity
        Q = np.array([query_vec])
        C = np.array(valid_chunk_vecs)
        scores = cosine_similarity(Q, C)[0]
        
        top_n = min(10, len(valid_chunks))
        top_indices = scores.argsort()[-top_n:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        filtered_chunks = [valid_chunks[i] for i in top_indices_sorted]
        
        # Update state with ALL embeddings (aligned with original valid chunks) for storage
        return {
            "filtered_chunks": filtered_chunks,
            "chunk_embeddings": valid_chunk_vecs,
            "chunks": valid_chunks # Update chunks to only valid ones
        }
        
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        return {"filtered_chunks": chunks[:10]}

async def extraction_node(state: FileParsingState):
    """Extract structured info using LLM."""
    if not state.get('success', True):
        return {}
        
    filtered_chunks = state.get('filtered_chunks', [])
    if not filtered_chunks:
        # Fallback to original chunks if ranking failed completely or not run
        filtered_chunks = state.get('chunks', [])[:10]
        
    context_text = "\n---\n".join(filtered_chunks)
    
    from core.config import Config
    llm = Config.get_utility_llm()
    
    # Prompt
    prompt = f"""
    你是专业的文档信息结构化提取助手，擅长从文本中精准提取指定信息并按固定格式输出，无遗漏、无冗余、不篡改原文信息。
    请从以下文档文本中，提取【标题、简介、详细描述、研究方向、技术栈、完成时间、预算、可提供的支持】这些核心信息，严格遵循以下要求：
    1. 标题：提取文档的核心主标题，若有副标题需一并提取，无则填“无”；
    2. 简介：提炼文档的核心主旨、创作目的、核心价值，控制在150字内，无则填“无”；
    3. 详细描述：提取文档的核心内容、关键细节、核心逻辑/框架，保留原文关键信息，无需缩写，无则填“无”；
    4. 研究方向：总结该需求项目的整体研究方向，用词语总结；
    5. 技术栈：分析该需求需要哪些技术技能，用词语总结；
    6. 完成时间：该需求项目的DDL，表示格式为“YYYY-MM-DD"，无则填“无”；
    7. 预算：提取文档中表示完成给需求项目时甲方可提供的资金支持，只提取数字（单位默认为万元，如“50”），不要包含“万元”等单位字样。若原文只有数字，默认其为万元，无则填“无”；
    8. 可提供的支持：除了资金外其余待遇、帮助辅助事宜等，无则填“无”
    9. 所有提取内容必须来自以下输入文本，不得凭空生成，原文无对应信息则统一填“无”；
    10. 严格按【指定输出格式】输出，不得修改格式、不得添加额外说明、不得换行混乱。

    【输入文档文本】
    {context_text}

    【指定输出格式】
    {{
      "title": "提取的具体标题内容",
      "brief": "提取的具体简介内容",
      "description": "提取的具体详细描述内容",
      "research_direction": "分析得到的研究方向总结",
      "skill": "分析所需的技术栈总结",
      "finish_time": "提取的具体结束时间期限",
      "budget": "提取的具体资金支持数字（单位万元），如'50'",
      "support_provided": "提取的具体额外支持"
    }}
    """
    
    try:
        # Use low temperature for extraction
        # Config.get_utility_llm() returns a configured instance. 
        # We can pass temperature override if the model supports bind or if we create a new instance.
        # ChatTongyi supports `temperature` in constructor or bind.
        # Since we use `llm.invoke`, we can't easily change temperature of the instance unless we re-instantiate or bind.
        # Let's try to bind if supported or just use default (which is usually low-ish or adjustable via params).
        # Assuming we can't easily change it without creating new LLM, we proceed. 
        # (User asked for 0.0-0.2, but qwen-turbo default might be higher).
        # Ideally: llm = ChatTongyi(..., temperature=0.1)
        
        # Re-instantiate for this specific task to ensure temperature constraint
        from langchain_community.chat_models import ChatTongyi
        llm_extraction = ChatTongyi(
            model='qwen-turbo', 
            api_key=Config.DASHSCOPE_API_KEY, 
            streaming=True,
            temperature=0.1 # Low temp as requested
        )
        
        response = await llm_extraction.ainvoke(prompt)
        content = response.content
        
        # Clean markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(content)
        
        # Post-process budget to ensure just number
        budget = str(data.get('budget', ''))
        if budget and budget != '无':
             # Try to extract number if LLM included units
             import re
             # Remove common units first to avoid confusion
             budget_clean = budget.replace("万元", "").replace("元", "").strip()
             # Extract first number (int or float)
             match = re.search(r'\d+(\.\d+)?', budget_clean)
             if match:
                 data['budget'] = match.group(0)
             else:
                 # If no number found, keep original (might be '面议' or similar)
                 pass
        
        return {
            "extracted_data": data, 
            "summary": data.get('brief', '无简介'), # Update summary for agent
            "success": True
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"success": False, "error": str(e), "extracted_data": {}}

# Removed summary_node in favor of extraction_node

async def vector_store_node(state: FileParsingState):
    """Store chunks to Milvus (project_raw_docs)."""
    # Only store if we have a draft_id (i.e. requirement created)
    # If draft_id is not present, we skip this step (Agent will handle creation first)
    # BUT, the design says this subgraph runs BEFORE or DURING chat.
    # Strategy: If no draft_id, we can't store to Milvus yet because we need project_id.
    # So we skip storage here. The Main Agent will call 'save_draft' tool, which can trigger storage 
    # OR we pass the parsed chunks back to Main Agent, and Main Agent calls a tool to save chunks later.
    
    # DECISION: This node is optional. If draft_id is passed, we store.
    draft_id = state.get('draft_id')
    if not draft_id:
        return {}
        
    chunks = state.get('chunks', [])
    chunk_embeddings = state.get('chunk_embeddings', [])
    
    # Check if we have embeddings from ranking_node
    if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
        logger.warning("Embeddings missing or mismatch in vector_store_node. Regenerating...")
        chunk_embeddings = []
        # Regenerate async
        chunk_embeddings = await EmbeddingService.aget_embeddings(chunks)
            
    try:
        collection = get_or_create_collection(COLLECTION_RAW_DOCS)
        
        # Batch insert preparation
        pids = []
        vectors = []
        contents = []
        indices = []
        
        for i, (chunk, vec) in enumerate(zip(chunks, chunk_embeddings)):
            if vec and len(vec) > 0 and not all(x == 0 for x in vec): # Valid vector
                pids.append(draft_id)
                vectors.append(vec)
                contents.append(chunk[:65535])
                indices.append(i)
        
        if not vectors:
            return {}
            
        final_data = [
            pids,
            vectors,
            contents,
            indices
        ]
        
        collection.insert(final_data)
        logger.info(f"Stored {len(vectors)} chunks for draft {draft_id}")
        return {"success": True}
        

    except Exception as e:
        logger.error(f"Vector storage failed: {e}")
        return {"success": False, "error": str(e)}

# --- Graph Construction ---
workflow = StateGraph(FileParsingState)

workflow.add_node("loader", loader_node)
workflow.add_node("cleaner", cleaner_node)
workflow.add_node("ranking", ranking_node)
workflow.add_node("extractor", extraction_node)
# Removed vector_store_node to prevent duplicate storage. 
# Storage is now handled exclusively by Backend Signals.

workflow.set_entry_point("loader")
workflow.add_edge("loader", "cleaner")
workflow.add_edge("cleaner", "ranking")
workflow.add_edge("ranking", "extractor")
workflow.add_edge("extractor", END) # Direct to END, skip vector_store

file_parsing_app = workflow.compile()
