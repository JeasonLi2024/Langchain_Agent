#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一 S2：批量调用平台文件解析工作流（file_parsing_graph + 标签推荐）
- 使用项目生产代码 graph/file_parsing_graph.py 的编译图，与 UI 链路完全一致
- 通过 astream(stream_mode="updates") 按实际节点名打点，差分得到各阶段耗时
- 输出对齐《实验评测方案》附录 A.1，另存 raw_text 供 S3-S6 标注复用
- 支持阶段级断点续跑（完整结果跳过；旧标签实现仅补跑标签，不重做解析）
- 默认按文件名顺序处理前 100 份 PDF（--limit 0 可解除限制）
- 默认 2 路并发跨 PDF 执行；每份内部节点顺序与生产流程一致（--workers 1 可做单请求基准）
- Milvus 使用本地 Lite 内嵌模式（.env: MILVUS_LITE_URI=./milvus.db，标签库已构建）
- Lite 标签索引为 L2 时仅在本脚本进程内转为高分相似度，不修改生产检索与工作流

用法（在 Langchain_Agent 目录下执行；本机需经 run_with_en0.py 绕过 AnyConnect 隧道，见实验手册 §3.3）：
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py                              # 按顺序处理前 100 份（默认 --limit 100）
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py --vectorize                  # 同时实测 T_vectorize（写入本地隔离 Milvus）
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py --skip-tags                  # 跳过标签推荐（无 Milvus 时）
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py --only-id 001 007            # 只跑指定编号
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py --tags-only                  # 只补跑标签（需已有解析结果）
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py --force                      # 强制重跑已存在的
    python scripts/run_with_en0.py scripts/exp1_batch_parse.py --workers 1                  # 串行基准；默认 workers=2
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANGCHAIN_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, LANGCHAIN_ROOT)

# The production retriever sorts scores in descending order. Remote production
# collections may expose similarity scores, while LangChain-created Milvus Lite
# collections default to raw L2 distance (smaller is better). This adapter is
# process-local to this experiment runner and restores the score contract the
# production ranking code expects without modifying the production workflow.
BATCH_TAG_SCORE_ADAPTER_VERSION = "exp1-lite-l2-to-similarity-v1"
TAG_COLLECTIONS = ("student_interests", "student_skills")
TAG_RETRIEVAL_SCORE_SEMANTICS = {}

# ── CWD 无关性保障：切到工程根（./.env 与 ./milvus.db 相对路径稳定），并显式加载 .env ──
os.chdir(LANGCHAIN_ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(LANGCHAIN_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv 为 pymilvus 依赖，正常必装

# ── 导入生产工作流（无需 Django 初始化，见实验手册 §1.3）──
from graph.file_parsing_graph import file_parsing_app          # 文件解析子图
from graph.tag_recommendation import recommend_tags_logic       # 生产标签推荐逻辑
from core.config import Config                                 # noqa: E402
from core.milvus_runtime import MilvusRuntime                   # noqa: E402


def _tag_implementation_fingerprint() -> str:
    """Fingerprint tag code, prompts, and non-secret model configuration."""
    digest = hashlib.sha256()
    for relative_path in (
        "graph/tag_recommendation.py",
        "tools/search_tools.py",
        "core/prompts.py",
        "core/config.py",
        "core/gateway_embeddings.py",
    ):
        path = os.path.join(LANGCHAIN_ROOT, relative_path)
        with open(path, "rb") as file:
            digest.update(relative_path.encode("utf-8"))
            digest.update(file.read())
    config = {
        "utility_model": Config.LLM_MODEL_UTILITY,
        "reasoning_model": Config.LLM_MODEL_REASONING,
        "embedding_model": Config.get_text_embedding_model(),
        "embedding_target": Config.get_text_embedding_target(),
        "embedding_base_url": Config.get_embedding_base_url(),
        "chat_base_url": Config.get_chat_base_url(),
        "embedding_dimension": Config.get_text_embedding_dimension(),
        "embedding_normalize": Config.TEXT_EMBEDDING_NORMALIZE,
        "embedding_query_instruction": Config.TEXT_EMBEDDING_QUERY_INSTRUCTION,
        "milvus_source": (
            Config.MILVUS_LITE_URI
            or f"{Config.MILVUS_HOST}:{Config.MILVUS_PORT}/{Config.MILVUS_DB_NAME or 'default'}"
        ),
        "batch_tag_score_adapter": BATCH_TAG_SCORE_ADAPTER_VERSION,
    }
    digest.update(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return "exp1.batch.tag_recommendation@sha256:" + digest.hexdigest()[:16]


TAG_IMPLEMENTATION = _tag_implementation_fingerprint()
TAGLESS_SEMANTIC_IMPLEMENTATION = "without-tags:v1"
# The batch-local score adapter changes candidate ordering, so no historical tag
# output is equivalent even when the production graph and prompt are unchanged.
LEGACY_EQUIVALENT_TAG_IMPLEMENTATIONS = set()
BATCH_SCHEMA_VERSION = 4

_milvus: MilvusRuntime | None = None
_milvus_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="exp1-milvus")


class _LiteL2SimilarityAdapter:
    """Expose Lite L2 results as descending similarity scores to production code."""

    _exp1_score_adapter = BATCH_TAG_SCORE_ADAPTER_VERSION

    def __init__(self, store):
        self._store = store

    @staticmethod
    def _to_similarity(distance: float) -> float:
        # Match langchain-milvus' own L2 relevance mapping. Embeddings are unit
        # normalized, so squared L2 is in [0, 4] and this mapping is monotonic.
        return 1.0 - float(distance) / 4.0

    @classmethod
    def _adapt_results(cls, results):
        return [(doc, cls._to_similarity(score)) for doc, score in results]

    def similarity_search_with_score(self, *args, **kwargs):
        return self._adapt_results(
            self._store.similarity_search_with_score(*args, **kwargs)
        )

    async def asimilarity_search_with_score(self, *args, **kwargs):
        return self._adapt_results(
            await self._store.asimilarity_search_with_score(*args, **kwargs)
        )

    def similarity_search_with_score_by_vector(self, *args, **kwargs):
        return self._adapt_results(
            self._store.similarity_search_with_score_by_vector(*args, **kwargs)
        )

    async def asimilarity_search_with_score_by_vector(self, *args, **kwargs):
        return self._adapt_results(
            await self._store.asimilarity_search_with_score_by_vector(*args, **kwargs)
        )

    def __getattr__(self, name):
        return getattr(self._store, name)


def _store_index_info(store) -> dict:
    """Read the collection's actual vector index instead of client defaults."""
    client = getattr(store, "client", None)
    collection_name = getattr(store, "collection_name", None)
    vector_field = getattr(store, "_vector_field", "vector")
    if client is None or not collection_name:
        raise RuntimeError("Milvus store does not expose client/collection metadata")

    index_names = client.list_indexes(collection_name=collection_name)
    matches = []
    for index_name in index_names:
        info = client.describe_index(
            collection_name=collection_name,
            index_name=index_name,
        )
        if info.get("field_name") == vector_field:
            matches.append(info)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one index for {collection_name}.{vector_field}, "
            f"found {len(matches)}"
        )

    metric_type = str(matches[0].get("metric_type") or "").upper()
    if not metric_type:
        raise RuntimeError(
            f"Milvus index {collection_name}.{vector_field} has no metric_type"
        )
    return {
        "metric_type": metric_type,
        "index_type": matches[0].get("index_type"),
        "field_name": vector_field,
    }


def _milvus_store_cache_key(collection_name: str):
    return (
        collection_name,
        Config.MILVUS_LITE_URI
        or f"{Config.MILVUS_HOST}:{Config.MILVUS_PORT}/{Config.MILVUS_DB_NAME}",
    )


def _configure_batch_tag_score_semantics() -> dict:
    """Adapt only this runner's Lite tag stores to production score semantics."""
    stores = {
        collection_name: Config.get_milvus_store(collection_name)
        for collection_name in TAG_COLLECTIONS
    }
    if not Config.MILVUS_LITE_URI:
        # A remote run already uses the production collection and score contract.
        # Do not inspect, reinterpret, or wrap it from the experiment runner.
        return {
            collection_name: {
                "source_metric": "production_managed",
                "source_index_type": "production_managed",
                "output_semantics": "production_native_descending_contract",
                "adapter": "none",
            }
            for collection_name in TAG_COLLECTIONS
        }

    # Validate every collection first. No adapter is installed until all checks pass,
    # so a bad second collection cannot leave the process in a partially wrapped state.
    index_info_by_collection = {
        collection_name: _store_index_info(store)
        for collection_name, store in stores.items()
    }
    for collection_name, index_info in index_info_by_collection.items():
        metric_type = index_info["metric_type"]
        if metric_type not in {"L2", "COSINE", "IP"}:
            raise RuntimeError(
                f"unsupported Lite tag metric {metric_type!r} for {collection_name}; "
                "cannot preserve production descending-score semantics"
            )
        if metric_type == "L2" and not Config.TEXT_EMBEDDING_NORMALIZE:
            raise RuntimeError(
                "Lite L2 score conversion requires TEXT_EMBEDDING_NORMALIZE=true"
            )

    semantics = {}
    for collection_name, store in stores.items():
        index_info = index_info_by_collection[collection_name]
        metric_type = index_info["metric_type"]
        adapter = "none"
        if metric_type == "L2":
            if not isinstance(store, _LiteL2SimilarityAdapter):
                store = _LiteL2SimilarityAdapter(store)
                Config._milvus_store_cache[
                    _milvus_store_cache_key(collection_name)
                ] = store
            adapter = BATCH_TAG_SCORE_ADAPTER_VERSION

        semantics[collection_name] = {
            "source_metric": metric_type,
            "source_index_type": index_info["index_type"],
            "output_semantics": "higher_is_more_similar",
            "adapter": adapter,
        }
    return semantics


async def _milvus_call(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_milvus_executor, partial(func, *args, **kwargs))


# 节点顺序与计时阶段的映射（对应方案 §1.2.1 计时点）
NODE_SEQUENCE = ["loader", "cleaner", "ranking", "extractor"]
STAGE_MAP = {
    "loader": "T_load",      # 文件加载与文本提取
    "cleaner": "T_clean",    # 文本清洗
    "ranking": "T_chunk",    # 分块与排序（ranker top-k）
    "extractor": "T_extract" # LLM 结构化字段提取
}


def extract_json_block(text: str) -> dict:
    """从 <thinking> + ```json``` 混合输出中提取 JSON（兼容裸 JSON）。"""
    if not text:
        return {}
    try:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return {}


def percentile(sorted_values, pct):
    """简单百分位（样本量 100 足够）。"""
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(pct / 100.0 * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _timing_total(timing: dict, include_tags: bool, include_vectorize: bool) -> float:
    stages = ["T_load", "T_clean", "T_chunk", "T_extract"]
    if include_tags:
        stages.append("T_tag")
    if include_vectorize:
        stages.append("T_vectorize")
    return round(sum(
        timing.get(stage) for stage in stages
        if isinstance(timing.get(stage), (int, float))
    ), 1)


async def run_parsing_graph(pdf_path: str):
    """Run the production graph and time actual node updates by node name."""
    input_state = {
        "file_path": os.path.abspath(pdf_path),
        "file_name": os.path.basename(pdf_path),
    }
    timing = {}
    current_state = dict(input_state)
    loader_state = {}
    previous = time.perf_counter()

    async for update in file_parsing_app.astream(input_state, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, node_output in update.items():
            finished = time.perf_counter()
            stage = STAGE_MAP.get(node_name, f"T_{node_name}")
            timing[stage] = round((finished - previous) * 1000, 1)
            previous = finished
            if isinstance(node_output, dict):
                current_state.update(node_output)
            if node_name == "loader":
                loader_state = dict(current_state)

    raw_text = (loader_state.get("chunks") or [""])[0] or ""
    success = bool(current_state.get("success"))
    error = current_state.get("error", "") if not success else ""
    extracted = current_state.get("extracted_data", {}) or {}
    chunks = current_state.get("chunks", []) or []
    chunk_embeddings = current_state.get("chunk_embeddings", []) or []
    if success and not raw_text.strip():
        success = False
        error = "empty text extracted (scanned/image PDF?)"
    return raw_text, extracted, timing, success, error, chunks, chunk_embeddings


async def run_tag_recommendation(extracted: dict):
    """调用生产标签推荐函数，不在批处理层重写关键词、召回、排序或 Prompt。"""
    started = time.perf_counter()
    try:
        raw = await recommend_tags_logic(
            description=extracted.get("description", ""),
            research_direction=extracted.get("research_direction", ""),
            skill=extracted.get("skill", ""),
            goal=extracted.get("goal", ""),
            expected_result=extracted.get("expected_result", ""),
        )
        parsed = extract_json_block(raw)
        if not parsed:
            raise ValueError("tag recommendation returned no parseable JSON")
        interest = [tag.get("name") for tag in (parsed.get("interest_tags") or []) if tag.get("name")]
        skill = [tag.get("name") for tag in (parsed.get("skill_tags") or []) if tag.get("name")]
        count_valid = len(interest) == 3 and len(skill) == 5
        return {
            "interest_tags": interest,
            "skill_tags": skill,
            "tag_summary": parsed.get("summary", ""),
            "raw_output": raw,
            "timing": round((time.perf_counter() - started) * 1000, 1),
            "count_valid": count_valid,
            # 数量不合规属于被测质量，不作为基础设施失败重试，否则会造成择优偏差。
            "warning": None if count_valid else f"unexpected tag count: {len(interest)}+{len(skill)}",
            "error": None,
        }
    except Exception as exc:
        return {
            "interest_tags": None,
            "skill_tags": None,
            "tag_summary": "",
            "raw_output": "",
            "timing": round((time.perf_counter() - started) * 1000, 1),
            "count_valid": False,
            "warning": None,
            "error": str(exc),
        }



def _build_semantic_text(extracted, tag_result) -> str:
    tags = list((tag_result or {}).get("interest_tags") or []) + list(
        (tag_result or {}).get("skill_tags") or []
    )
    return (
        f"Title: {extracted.get('title', '')}\nBrief: {extracted.get('brief', '')}\n"
        f"Description: {extracted.get('description', '')}\nGoal: {extracted.get('goal', '')}\n"
        f"Expected Result: {extracted.get('expected_result', '')}\n"
        f"Contact Person: {extracted.get('contact_person', '')}\n"
        f"Contact Info: {extracted.get('contact_info', '')}\n"
        f"Tags: {', '.join(tags)}"
    )


def _fake_requirement_id(pdf_id: str) -> int:
    try:
        return 900000 + int(re.sub(r"\D", "", pdf_id) or 0)
    except ValueError:
        return 900000


async def run_vectorize(pdf_id, extracted, tag_result, chunks, chunk_embeddings):
    """Write both project vector collections through the process-wide Milvus client."""
    started = time.perf_counter()
    fake_req_id = _fake_requirement_id(pdf_id)
    try:
        if _milvus is None:
            raise RuntimeError("Milvus runtime is not initialized")
        full_text = _build_semantic_text(extracted, tag_result)
        semantic_vectors = await Config.get_embeddings().aembed_documents([full_text])
        semantic_vector = semantic_vectors[0] if semantic_vectors else []
        # 全部 Milvus Lite 写入经单线程 executor 串行化；同一 fake_req_id 先删后写，
        # 因此断点续跑不会产生重复向量。
        inserted_chunks, semantic_ok = await _milvus_call(
            _milvus.replace_project_vectors,
            fake_req_id,
            chunks,
            chunk_embeddings,
            full_text,
            semantic_vector,
        )
        return {
            "timing": round((time.perf_counter() - started) * 1000, 1),
            "inserted_chunks": inserted_chunks,
            "semantic_ok": semantic_ok,
            "fake_req_id": fake_req_id,
            "semantic_tag_implementation": (
                TAG_IMPLEMENTATION if tag_result is not None else TAGLESS_SEMANTIC_IMPLEMENTATION
            ),
            "retries": 0,
            "error": None,
        }
    except Exception as exc:
        return {
            "timing": round((time.perf_counter() - started) * 1000, 1),
            "inserted_chunks": 0,
            "semantic_ok": False,
            "fake_req_id": fake_req_id,
            "semantic_tag_implementation": (
                TAG_IMPLEMENTATION if tag_result is not None else TAGLESS_SEMANTIC_IMPLEMENTATION
            ),
            "retries": 0,
            "error": str(exc),
        }


async def run_semantic_vectorize(pdf_id, extracted, tag_result):
    """Refresh only the tag-dependent semantic vector after a tag-only resume."""
    started = time.perf_counter()
    fake_req_id = _fake_requirement_id(pdf_id)
    try:
        if _milvus is None:
            raise RuntimeError("Milvus runtime is not initialized")
        full_text = _build_semantic_text(extracted, tag_result)
        semantic_vectors = await Config.get_embeddings().aembed_documents([full_text])
        semantic_vector = semantic_vectors[0] if semantic_vectors else []
        semantic_ok = await _milvus_call(
            _milvus.replace_semantic_vector,
            fake_req_id,
            full_text,
            semantic_vector,
        )
        return {
            "timing": round((time.perf_counter() - started) * 1000, 1),
            "semantic_ok": semantic_ok,
            "fake_req_id": fake_req_id,
            "semantic_tag_implementation": (
                TAG_IMPLEMENTATION if tag_result is not None else TAGLESS_SEMANTIC_IMPLEMENTATION
            ),
            "error": None if semantic_ok else "semantic embedding was empty",
        }
    except Exception as exc:
        return {
            "timing": round((time.perf_counter() - started) * 1000, 1),
            "semantic_ok": False,
            "fake_req_id": fake_req_id,
            "semantic_tag_implementation": (
                TAG_IMPLEMENTATION if tag_result is not None else TAGLESS_SEMANTIC_IMPLEMENTATION
            ),
            "error": str(exc),
        }


def _write_json_atomic(path: str, payload: dict) -> None:
    """同目录原子替换，避免中断时留下半截 JSON 并破坏断点续跑。"""
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _write_text_atomic(path: str, text: str) -> None:
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _write_failure_record(out_dir: str, pdf_id: str, error: str) -> None:
    """Record a failed attempt without destroying reusable results from earlier stages."""
    out_json = os.path.join(out_dir, f"{pdf_id}_parsed.json")
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as file:
                existing = json.load(file)
            if _parse_is_complete(existing, out_dir):
                existing["last_attempt_error"] = error
                existing["last_attempt_success"] = False
                _write_json_atomic(out_json, existing)
                return
        except (OSError, json.JSONDecodeError):
            pass

    stages = ("T_load", "T_clean", "T_chunk", "T_extract", "T_tag", "T_vectorize", "T_total")
    record = {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "tag_implementation": None,
        "pdf_id": pdf_id,
        "success": False,
        "parse_success": False,
        "tag_success": False,
        "vectorize_success": False,
        "error": error,
        "last_attempt_error": error,
        "last_attempt_success": False,
        "timing": {stage: None for stage in stages},
    }
    _write_json_atomic(out_json, record)


def _parse_is_complete(record: dict, out_dir: str | None = None) -> bool:
    timing = record.get("timing") or {}
    required_stages = ("T_load", "T_clean", "T_chunk", "T_extract")
    pdf_id = record.get("pdf_id")
    raw_text_ok = bool(
        out_dir is None
        or (
            pdf_id
            and os.path.isfile(os.path.join(out_dir, f"{pdf_id}_raw_text.txt"))
            and os.path.getsize(os.path.join(out_dir, f"{pdf_id}_raw_text.txt")) > 0
        )
    )
    return bool(
        record.get("parse_success", record.get("success", False))
        and all(isinstance(timing.get(stage), (int, float)) for stage in required_stages)
        and record.get("title")
        and raw_text_ok
    )


def _tags_are_complete(record: dict) -> bool:
    return bool(
        record.get("tag_implementation") == TAG_IMPLEMENTATION
        and record.get("tag_retrieval_score_semantics") == TAG_RETRIEVAL_SCORE_SEMANTICS
        and record.get("tag_success") is True
        and isinstance(record.get("interest_tags"), list)
        and isinstance(record.get("skill_tags"), list)
        and not record.get("tag_error")
    )


def _raw_vectors_are_complete(record: dict) -> bool:
    detail = record.get("vectorize_detail") or {}
    return int(detail.get("inserted_chunks") or 0) > 0


def _vectors_are_complete(record: dict, expected_semantic_implementation: str) -> bool:
    detail = record.get("vectorize_detail") or {}
    return bool(
        not detail.get("error")
        and detail.get("semantic_ok")
        and _raw_vectors_are_complete(record)
        and detail.get("semantic_tag_implementation") == expected_semantic_implementation
    )


def _record_is_complete(record: dict, args, out_dir: str | None = None) -> bool:
    if not _parse_is_complete(record, out_dir):
        return False
    if not args.skip_tags and not _tags_are_complete(record):
        return False
    expected_semantic_implementation = (
        TAGLESS_SEMANTIC_IMPLEMENTATION if args.skip_tags else TAG_IMPLEMENTATION
    )
    if args.vectorize and not _vectors_are_complete(record, expected_semantic_implementation):
        return False
    return True


def build_parsed_record(pdf_id, extracted, tag_result, timing, parse_success, error, vec_result=None):
    """Build the result and distinguish parse success from end-to-end success."""
    tag_success = None if tag_result is None else not tag_result.get("error")
    vectorize_success = (
        None
        if vec_result is None
        else (
            not vec_result.get("error")
            and bool(vec_result.get("semantic_ok"))
            and int(vec_result.get("inserted_chunks") or 0) > 0
        )
    )
    pipeline_success = bool(
        parse_success
        and tag_success is not False
        and vectorize_success is not False
    )
    errors = [
        item for item in (
            error,
            tag_result.get("error") if tag_result else None,
            vec_result.get("error") if vec_result else None,
        )
        if item
    ]
    rec = {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "tag_implementation": TAG_IMPLEMENTATION if tag_result is not None else None,
        "tag_retrieval_score_semantics": (
            dict(TAG_RETRIEVAL_SCORE_SEMANTICS) if tag_result is not None else None
        ),
        "pdf_id": pdf_id,
        "success": pipeline_success,
        "parse_success": bool(parse_success),
        "tag_success": tag_success if tag_result is not None else None,
        "vectorize_success": vectorize_success if vec_result is not None else None,
        "title": extracted.get("title", ""),
        "brief": extracted.get("brief", ""),
        "description": extracted.get("description", ""),
        "goal": extracted.get("goal", ""),
        "expected_result": extracted.get("expected_result", ""),
        "budget": extracted.get("budget", ""),
        "people_count": None,
        "support_provided": extracted.get("support_provided", ""),
        "contact_person": extracted.get("contact_person", ""),
        "contact_info": extracted.get("contact_info", ""),
        "interest_tags": tag_result.get("interest_tags") if tag_result else None,
        "skill_tags": tag_result.get("skill_tags") if tag_result else None,
        "research_direction": extracted.get("research_direction", ""),
        "skill": extracted.get("skill", ""),
        "finish_time": extracted.get("finish_time", ""),
        "tag_summary": tag_result.get("tag_summary", "") if tag_result else "",
        "tag_count_valid": tag_result.get("count_valid") if tag_result else None,
        "tag_warning": tag_result.get("warning") if tag_result else None,
        "tag_error": tag_result.get("error") if tag_result else None,
        "error": "; ".join(errors) or None,
        "timing": timing,
    }
    if tag_result:
        rec["timing"]["T_tag"] = tag_result["timing"]
    if vec_result is not None:
        rec["timing"]["T_vectorize"] = vec_result["timing"]
        rec["vectorize_detail"] = {
            "inserted_chunks": vec_result["inserted_chunks"],
            "semantic_ok": vec_result["semantic_ok"],
            "fake_req_id": vec_result["fake_req_id"],
            "semantic_tag_implementation": vec_result.get("semantic_tag_implementation"),
            "retries": vec_result.get("retries", 0),
            "error": vec_result["error"],
        }
    else:
        rec["timing"].setdefault("T_vectorize", None)
    rec["timing"]["T_total"] = round(sum(
        value for value in rec["timing"].values() if isinstance(value, (int, float))
    ), 1)
    return rec


async def process_pdf(pdf_path, pdf_id, out_dir, args):
    out_json = os.path.join(out_dir, f"{pdf_id}_parsed.json")
    out_txt = os.path.join(out_dir, f"{pdf_id}_raw_text.txt")
    existing = None

    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as file:
                existing = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[retry] {pdf_id} 结果文件不可用，将安全重跑：{exc}")

    # One-time migration for results produced earlier in this same audit. The only
    # subsequent tag-code change tightened failure reporting; successful output is equivalent.
    if (
        existing is not None
        and existing.get("tag_implementation") in LEGACY_EQUIVALENT_TAG_IMPLEMENTATIONS
        and existing.get("tag_success") is True
        and isinstance(existing.get("interest_tags"), list)
        and isinstance(existing.get("skill_tags"), list)
        and not existing.get("tag_error")
    ):
        existing["tag_implementation"] = TAG_IMPLEMENTATION
        existing["batch_schema_version"] = BATCH_SCHEMA_VERSION
        _write_json_atomic(out_json, existing)
        print(f"[migrate] {pdf_id} 已验证的生产标签结果升级为源码指纹")

    if existing is not None and not args.force and not args.tags_only:
        if _record_is_complete(existing, args, out_dir):
            print(f"[skip] {pdf_id} 已有完整健康结果")
            return None

        # 已完成的生产解析阶段不重复调用；旧脚本改写过标签召回时，仅按生产函数补跑标签。
        can_resume_tags = (
            _parse_is_complete(existing, out_dir)
            and not args.skip_tags
            and not _tags_are_complete(existing)
            and (not args.vectorize or _raw_vectors_are_complete(existing))
        )
        if can_resume_tags:
            action = "仅按生产逻辑补跑标签"
            if args.vectorize:
                action += "并刷新标签相关语义向量"
            print(f"[resume] {pdf_id} 保留解析结果，{action}")
            extracted = {key: existing.get(key, "") for key in (
                "title", "brief", "description", "research_direction", "skill",
                "goal", "expected_result", "budget", "support_provided",
                "contact_person", "contact_info", "finish_time",
            )}
            tag_result = await run_tag_recommendation(extracted)
            timing = dict(existing.get("timing") or {})
            timing["T_tag"] = tag_result["timing"]
            detail = dict(existing.get("vectorize_detail") or {})
            if tag_result.get("error"):
                detail["semantic_tag_implementation"] = None
            elif args.vectorize:
                semantic_result = await run_semantic_vectorize(pdf_id, extracted, tag_result)
                timing["T_vectorize"] = semantic_result["timing"]
                detail.update({
                    "semantic_ok": semantic_result["semantic_ok"],
                    "fake_req_id": semantic_result["fake_req_id"],
                    "semantic_tag_implementation": semantic_result["semantic_tag_implementation"],
                    "error": semantic_result["error"],
                })
            else:
                # Existing project semantic vectors contain the previous labels. Mark them stale
                # so a later --vectorize run cannot mistake them for current output.
                detail["semantic_tag_implementation"] = None
            timing["T_total"] = _timing_total(
                timing,
                include_tags=True,
                include_vectorize=args.vectorize,
            )
            existing.update({
                "batch_schema_version": BATCH_SCHEMA_VERSION,
                "tag_implementation": TAG_IMPLEMENTATION,
                "tag_retrieval_score_semantics": dict(TAG_RETRIEVAL_SCORE_SEMANTICS),
                "interest_tags": tag_result["interest_tags"],
                "skill_tags": tag_result["skill_tags"],
                "tag_summary": tag_result["tag_summary"],
                "tag_count_valid": tag_result["count_valid"],
                "tag_warning": tag_result["warning"],
                "tag_error": tag_result["error"],
                "tag_success": not tag_result["error"],
                "vectorize_detail": detail or None,
                "vectorize_success": (
                    _vectors_are_complete(
                        {"vectorize_detail": detail},
                        TAG_IMPLEMENTATION,
                    )
                    if detail else None
                ),
                "timing": timing,
                "last_attempt_success": not tag_result["error"],
                "last_attempt_error": tag_result["error"],
            })
            existing["success"] = bool(
                existing.get("parse_success", existing.get("success", False))
                and existing["tag_success"]
                and (
                    not args.vectorize
                    or _vectors_are_complete(existing, TAG_IMPLEMENTATION)
                )
            )
            vector_error = (existing.get("vectorize_detail") or {}).get("error") if args.vectorize else None
            existing["error"] = tag_result["error"] or vector_error
            existing["last_attempt_success"] = existing["error"] is None
            existing["last_attempt_error"] = existing["error"]
            _write_json_atomic(out_json, existing)
            existing["_run_action"] = "tag_and_semantic_resume" if args.vectorize else "tag_resume"
            print(f"[tags] {pdf_id}: 兴趣{len(existing['interest_tags'] or [])} "
                  f"技能{len(existing['skill_tags'] or [])} T_tag={timing['T_tag']}ms")
            return existing

        can_resume_vector = (
            args.vectorize
            and _parse_is_complete(existing, out_dir)
            and (args.skip_tags or _tags_are_complete(existing))
            and _raw_vectors_are_complete(existing)
        )
        if can_resume_vector:
            print(f"[resume] {pdf_id} 保留解析/标签/原始块向量，仅刷新语义向量")
            extracted = {key: existing.get(key, "") for key in (
                "title", "brief", "description", "research_direction", "skill",
                "goal", "expected_result", "budget", "support_provided",
                "contact_person", "contact_info", "finish_time",
            )}
            tag_result = None if args.skip_tags else {
                "interest_tags": existing.get("interest_tags") or [],
                "skill_tags": existing.get("skill_tags") or [],
            }
            semantic_result = await run_semantic_vectorize(pdf_id, extracted, tag_result)
            detail = dict(existing.get("vectorize_detail") or {})
            detail.update({
                "semantic_ok": semantic_result["semantic_ok"],
                "fake_req_id": semantic_result["fake_req_id"],
                "semantic_tag_implementation": semantic_result["semantic_tag_implementation"],
                "error": semantic_result["error"],
            })
            timing = dict(existing.get("timing") or {})
            timing["T_vectorize"] = semantic_result["timing"]
            timing["T_total"] = _timing_total(
                timing,
                include_tags=not args.skip_tags,
                include_vectorize=True,
            )
            existing.update({
                "batch_schema_version": BATCH_SCHEMA_VERSION,
                "vectorize_detail": detail,
                "vectorize_success": _vectors_are_complete(
                    {"vectorize_detail": detail},
                    TAGLESS_SEMANTIC_IMPLEMENTATION if args.skip_tags else TAG_IMPLEMENTATION,
                ),
                "timing": timing,
                "last_attempt_success": not semantic_result["error"],
                "last_attempt_error": semantic_result["error"],
            })
            existing["success"] = bool(
                _parse_is_complete(existing, out_dir)
                and (args.skip_tags or _tags_are_complete(existing))
                and existing["vectorize_success"]
            )
            existing["error"] = semantic_result["error"]
            _write_json_atomic(out_json, existing)
            existing["_run_action"] = "semantic_vector_resume"
            return existing

        print(f"[retry] {pdf_id} 现有结果缺少不可复用阶段，自动重跑")

    if args.tags_only:
        if not os.path.exists(out_json):
            print(f"[skip] {pdf_id} 无已有解析结果，无法仅补标签")
            return None
        with open(out_json, "r", encoding="utf-8") as file:
            old = json.load(file)
        tags_complete = _tags_are_complete(old)
        if tags_complete and not args.force:
            print(f"[skip] {pdf_id} 已有完整 3+5 标签")
            return None
        extracted = {key: old.get(key, "") for key in (
            "title", "brief", "description", "research_direction", "skill",
            "goal", "expected_result", "budget", "support_provided",
            "contact_person", "contact_info", "finish_time",
        )}
        tag_result = await run_tag_recommendation(extracted)
        timing = dict(old.get("timing") or {})
        timing["T_tag"] = tag_result["timing"]
        detail = dict(old.get("vectorize_detail") or {})
        if tag_result.get("error"):
            detail["semantic_tag_implementation"] = None
        elif args.vectorize and _raw_vectors_are_complete(old):
            semantic_result = await run_semantic_vectorize(pdf_id, extracted, tag_result)
            timing["T_vectorize"] = semantic_result["timing"]
            detail.update({
                "semantic_ok": semantic_result["semantic_ok"],
                "fake_req_id": semantic_result["fake_req_id"],
                "semantic_tag_implementation": semantic_result["semantic_tag_implementation"],
                "error": semantic_result["error"],
            })
        else:
            detail["semantic_tag_implementation"] = None
        timing["T_total"] = _timing_total(
            timing,
            include_tags=True,
            include_vectorize=args.vectorize,
        )
        old.update({
            "batch_schema_version": BATCH_SCHEMA_VERSION,
            "tag_implementation": TAG_IMPLEMENTATION,
            "tag_retrieval_score_semantics": dict(TAG_RETRIEVAL_SCORE_SEMANTICS),
            "interest_tags": tag_result["interest_tags"],
            "skill_tags": tag_result["skill_tags"],
            "tag_summary": tag_result["tag_summary"],
            "tag_count_valid": tag_result["count_valid"],
            "tag_warning": tag_result["warning"],
            "tag_error": tag_result["error"],
            "tag_success": not tag_result["error"],
            "vectorize_detail": detail or None,
            "vectorize_success": (
                _vectors_are_complete({"vectorize_detail": detail}, TAG_IMPLEMENTATION)
                if detail else None
            ),
            "timing": timing,
            "last_attempt_success": not tag_result["error"],
            "last_attempt_error": tag_result["error"],
        })
        old["parse_success"] = old.get("parse_success", old.get("success", False))
        old["success"] = bool(
            old["parse_success"]
            and old["tag_success"]
            and (not args.vectorize or old.get("vectorize_success") is True)
        )
        vector_error = (old.get("vectorize_detail") or {}).get("error") if args.vectorize else None
        old["error"] = tag_result["error"] or vector_error
        old["last_attempt_success"] = old["error"] is None
        old["last_attempt_error"] = old["error"]
        _write_json_atomic(out_json, old)
        old["_run_action"] = "tag_and_semantic_resume" if args.vectorize else "tag_resume"
        print(f"[tags] {pdf_id}: 兴趣{len(old['interest_tags'] or [])} "
              f"技能{len(old['skill_tags'] or [])} T_tag={timing['T_tag']}ms")
        return old

    print(f"[run ] {pdf_id} ...", flush=True)
    raw_text, extracted, timing, parse_success, error, chunks, chunk_embeddings = (
        await run_parsing_graph(pdf_path)
    )
    _write_text_atomic(out_txt, raw_text)

    tag_result = None
    if parse_success and not args.skip_tags:
        tag_result = await run_tag_recommendation(extracted)
    vec_result = None
    if parse_success and args.vectorize:
        vec_result = await run_vectorize(
            pdf_id, extracted, tag_result, chunks, chunk_embeddings
        )
    rec = build_parsed_record(
        pdf_id, extracted, tag_result, timing, parse_success, error, vec_result
    )
    rec["last_attempt_success"] = rec["success"]
    rec["last_attempt_error"] = rec["error"]
    _write_json_atomic(out_json, rec)
    rec["_run_action"] = "full_pipeline"

    status = "OK " if rec["success"] else "FAIL"
    print(
        f"[{status}] {pdf_id}: T_total={rec['timing']['T_total']}ms "
        f"(load={timing.get('T_load')}, clean={timing.get('T_clean')}, "
        f"chunk={timing.get('T_chunk')}, extract={timing.get('T_extract')}, "
        f"tag={rec['timing'].get('T_tag')}, "
        f"vectorize={rec['timing'].get('T_vectorize')})"
        + (f" ERROR: {rec['error']}" if rec.get("error") else "")
    )
    return rec

def preflight_check(skip_tags, vectorize):
    """预热标签存储，并仅在 Lite 模式适配生产代码所需的分数语义。"""
    global _milvus, TAG_RETRIEVAL_SCORE_SEMANTICS
    mode = f"Lite embedded ({Config.MILVUS_LITE_URI})" if Config.MILVUS_LITE_URI else (
        f"remote {Config.MILVUS_HOST}:{Config.MILVUS_PORT}/{Config.MILVUS_DB_NAME or 'default'}"
    )
    print(f"[preflight] Milvus mode: {mode}")
    if not skip_tags:
        # recommend_tags_logic/retrieve_tags/Prompt 均保持生产实现不变。仅当实验使用的
        # Lite 标签索引返回 L2 距离时，在本进程缓存中包一层分数适配器，使生产代码
        # 继续按“分数越大越相关”的契约排序。
        TAG_RETRIEVAL_SCORE_SEMANTICS = _configure_batch_tag_score_semantics()
        for collection_name, score_info in TAG_RETRIEVAL_SCORE_SEMANTICS.items():
            print(
                f"[preflight] {collection_name}: metric={score_info['source_metric']} "
                f"adapter={score_info['adapter']} output={score_info['output_semantics']}"
            )
        print("[preflight] production tag workflow initialized with batch-local score semantics")
    if vectorize:
        _milvus = MilvusRuntime(rpc_timeout=30)
        _milvus.initialize(require_tags=False, require_projects=True)
        print("[preflight] dedicated project-vector writer initialized")
    if skip_tags and not vectorize:
        print("[preflight] Milvus is not needed (--skip-tags without --vectorize)")


def summarize(out_dir, args, target_ids=None, run_actions=None):
    """Summarize the requested set and measure throughput from this invocation only."""
    target_ids = list(dict.fromkeys(target_ids or []))
    run_actions = dict(run_actions or {})
    processed_ids = set(run_actions)
    records_by_id = {}
    unreadable = {}
    for pdf_id in target_ids:
        path = os.path.join(out_dir, f"{pdf_id}_parsed.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file:
                record = json.load(file)
            records_by_id[pdf_id] = record
        except (OSError, json.JSONDecodeError) as exc:
            unreadable[pdf_id] = str(exc)

    records = [records_by_id[pdf_id] for pdf_id in target_ids if pdf_id in records_by_id]
    complete = [
        record for record in records
        if _record_is_complete(record, args, out_dir)
    ]
    complete_ids = {record.get("pdf_id") for record in complete}
    failed = []
    for pdf_id in target_ids:
        record = records_by_id.get(pdf_id)
        if pdf_id in complete_ids:
            continue
        failed.append({
            "pdf_id": pdf_id,
            "parse_success": (
                record.get("parse_success", record.get("success", False)) if record else False
            ),
            "tag_success": record.get("tag_success") if record else None,
            "vectorize_success": record.get("vectorize_success") if record else None,
            "tag_implementation": record.get("tag_implementation") if record else None,
            "error": (
                unreadable.get(pdf_id)
                or (record.get("error") if record else None)
                or ("result file missing" if record is None else "requested stages incomplete or stale")
            ),
        })
    stages = ["T_load", "T_clean", "T_chunk", "T_extract"]
    if not args.skip_tags:
        stages.append("T_tag")
    if args.vectorize:
        stages.append("T_vectorize")
    stages.append("T_total")
    summary = {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "tag_implementation": None if args.skip_tags else TAG_IMPLEMENTATION,
        "tag_retrieval_score_semantics": (
            None if args.skip_tags else dict(TAG_RETRIEVAL_SCORE_SEMANTICS)
        ),
        "requested_vectorize": bool(args.vectorize),
        "workers": args.workers,
        "wall_clock_seconds": round(time.time() - args.started_at, 1) if hasattr(args, "started_at") else None,
        "stage_semantics": {
            "T_load": "loader_node: PDF load and raw text extraction",
            "T_clean": "cleaner_node: regex cleaning, splitting, and short-chunk filtering",
            "T_chunk": "ranking_node: query/chunk embedding, cosine ranking, and top-10 selection",
            "T_extract": "extraction_node: LLM structured extraction",
            "T_tag": "recommend_tags_logic: keyword extraction, Milvus retrieval, and LLM selection",
            "T_vectorize": "experimental synchronous persistence; production runs after requirement save",
            "T_total": "sum of all enabled timed stages, including T_vectorize when requested",
        },
        "total": len(target_ids),
        "success": len(complete),
        "failed": len(failed),
        "processed_this_run": len(processed_ids),
        "skipped_historical": len(target_ids) - len(processed_ids),
        "run_action_counts": {
            action: sum(current == action for current in run_actions.values())
            for action in sorted(set(run_actions.values()))
        },
        "parse_success": sum(
            bool(record.get("parse_success", record.get("success", False))) for record in records
        ),
        "tag_success": (
            None if args.skip_tags
            else sum(record.get("tag_success") is True for record in records)
        ),
        "tag_count_valid": (
            None if args.skip_tags
            else sum(record.get("tag_count_valid") is True for record in records)
        ),
        "tag_count_warning_ids": [
            record.get("pdf_id") for record in records
            if record.get("tag_success") is True and record.get("tag_count_valid") is False
        ],
        "vectorize_success": (
            sum(record.get("vectorize_success") is True for record in records)
            if args.vectorize else None
        ),
        "failed_detail": failed,
        "stages_ms": {},
        "throughput": {},
    }
    for stage in stages:
        if stage == "T_total":
            values = sorted(
                _timing_total(
                    record.get("timing") or {},
                    include_tags=not args.skip_tags,
                    include_vectorize=args.vectorize,
                )
                for record in complete
            )
        else:
            values = sorted(
                record.get("timing", {}).get(stage)
                for record in complete
                if isinstance(record.get("timing", {}).get(stage), (int, float))
            )
        if values:
            summary["stages_ms"][stage] = {
                "mean": round(sum(values) / len(values), 1),
                "median": values[len(values) // 2],
                "p95": percentile(values, 95),
                "min": values[0],
                "max": values[-1],
                "n": len(values),
            }
    wall_clock_seconds = summary.get("wall_clock_seconds")
    processed_complete_ids = complete_ids & processed_ids
    completed_actions = {
        pdf_id: run_actions[pdf_id]
        for pdf_id in processed_complete_ids
        if run_actions[pdf_id] != "failed"
    }
    if completed_actions and isinstance(wall_clock_seconds, (int, float)) and wall_clock_seconds > 0:
        summary["throughput"] = {
            "basis": "completed work items from this invocation",
            "completed_this_run": len(completed_actions),
            "action_counts": {
                action: sum(current == action for current in completed_actions.values())
                for action in sorted(set(completed_actions.values()))
            },
            "completed_per_minute": round(len(completed_actions) * 60 / wall_clock_seconds, 3),
        }
        if set(completed_actions.values()) == {"full_pipeline"}:
            summary["throughput"]["full_pipeline_minutes_for_100_at_same_rate"] = round(
                wall_clock_seconds * 100 / len(completed_actions) / 60,
                1,
            )
    _write_json_atomic(os.path.join(out_dir, "exp1_timing_summary.json"), summary)
    print("\n===== S2 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    total = len(target_ids)
    passed = bool(total and len(complete) == total)
    print(
        f"\nCP2 校验：{len(complete)}/{total} 全链路成功"
        + ("  ✅ 通过" if passed else "  ❌ 未通过，需补跑失败项")
    )

async def main():
    global _milvus
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100,
                        help="按文件名顺序只处理前 N 份（默认 100；0 表示不限制）")
    parser.add_argument("--pdf-dir", default=os.path.join(LANGCHAIN_ROOT, "..", "data", "exp1_pdfs"))
    parser.add_argument("--out-dir", default=os.path.join(LANGCHAIN_ROOT, "..", "data", "exp1_results"))
    parser.add_argument("--skip-tags", action="store_true", help="跳过标签推荐")
    parser.add_argument("--vectorize", action="store_true", help="实测 T_vectorize 并写入隔离 Milvus")
    parser.add_argument("--tags-only", action="store_true", help="仅补跑标签")
    parser.add_argument("--only-id", nargs="*", default=[], help="只处理指定 pdf_id")
    parser.add_argument("--force", action="store_true", help="强制重跑已存在结果")
    parser.add_argument("--sleep", type=float, default=2.0, help="每个工作槽处理完成后的冷却秒数")
    parser.add_argument("--workers", type=int, default=2,
                        help="并发处理 PDF 数（默认 2；单请求基准测试建议设 1）")
    parser.add_argument("--watchdog", type=float, default=600,
                        help="单份 PDF 总超时秒数（底层 HTTP/gRPC 另有独立 deadline）")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must be between 1 and 4")

    pdf_dir = os.path.abspath(args.pdf_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    preflight_check(args.skip_tags, args.vectorize)

    pdfs = sorted(fn for fn in os.listdir(pdf_dir) if fn.lower().endswith(".pdf"))
    if args.only_id:
        pdfs = [fn for fn in pdfs if fn.split("_")[0] in args.only_id]
    elif args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    print(f"待处理 PDF：{len(pdfs)} 份 | 并发：{args.workers} | 输出目录：{out_dir}")
    started = time.time()
    args.started_at = started
    semaphore = asyncio.Semaphore(args.workers)
    run_actions = {}

    async def run_one(fn):
        async with semaphore:
            pdf_id = fn.split("_")[0]
            processed = False
            try:
                result = await asyncio.wait_for(
                    process_pdf(os.path.join(pdf_dir, fn), pdf_id, out_dir, args),
                    timeout=args.watchdog,
                )
                processed = result is not None
                if processed:
                    run_actions[pdf_id] = result.get("_run_action", "unknown")
            except asyncio.TimeoutError:
                processed = True
                error = f"watchdog timeout ({args.watchdog:.0f}s)"
                print(f"[FAIL] {pdf_id}: {error}")
                _write_failure_record(out_dir, pdf_id, error)
                run_actions[pdf_id] = "failed"
            except Exception as exc:
                processed = True
                error = f"unhandled processing error: {exc}"
                print(f"[FAIL] {pdf_id}: {error}")
                _write_failure_record(out_dir, pdf_id, error)
                run_actions[pdf_id] = "failed"
            if processed and args.sleep > 0:
                await asyncio.sleep(args.sleep)

    try:
        await asyncio.gather(*(run_one(fn) for fn in pdfs))
        summarize(
            out_dir,
            args,
            [fn.split("_")[0] for fn in pdfs],
            run_actions,
        )
        print(f"\n总耗时 {round((time.time() - started) / 60, 1)} 分钟")
    finally:
        if _milvus is not None:
            await _milvus_call(_milvus.close)
            _milvus = None
        _milvus_executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    asyncio.run(main())
