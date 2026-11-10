"""Project settings. No model request happens when this module is imported."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v1"
    dim: int = 1536
    llm_model: str = "qwen-plus"
    milvus_uri: str = "http://127.0.0.1:19530"
    paper: Path = ROOT / "papers" / "paper.pdf"
    chunk_size: int = 500
    overlap: int = 50
    max_l2: float | None = None
    rerank_enabled: bool = True
    rerank_model: str = "qwen3-rerank"
    rerank_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    candidate_k: int = 20


def settings():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    threshold = os.getenv("MAX_L2_DISTANCE", "").strip()
    path = Path(os.getenv("PAPER_PATH", "papers/paper.pdf")).expanduser()
    cfg = Settings(
        key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        base_url=os.getenv("BAILIAN_BASE_URL", Settings.base_url).strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v1"),
        dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
        milvus_uri=os.getenv("MILVUS_URI", Settings.milvus_uri),
        paper=(path if path.is_absolute() else ROOT / path).resolve(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        max_l2=float(threshold) if threshold else None,
        rerank_enabled=os.getenv("RERANK_ENABLED", "true").strip().lower() == "true",
        rerank_model=os.getenv("RERANK_MODEL", Settings.rerank_model).strip(),
        rerank_url=os.getenv("RERANK_URL", Settings.rerank_url).strip(),
        candidate_k=int(os.getenv("RETRIEVAL_CANDIDATES", "20")),
    )
    validate(cfg)
    return cfg


def validate(cfg):
    if not 0 <= cfg.overlap < cfg.chunk_size <= 1200 or cfg.dim <= 0:
        raise ValueError("参数需要满足 0 <= overlap < chunk_size <= 1200，向量维度必须为正。")
    if cfg.max_l2 is not None and (not math.isfinite(cfg.max_l2) or cfg.max_l2 < 0):
        raise ValueError("L2 距离上限必须是有限的非负数。")
    if not 10 <= cfg.candidate_k <= 50:
        raise ValueError("候选片段数需要在 10 到 50 之间。")


def preview_settings(cfg, size, overlap):
    for value in (size, overlap):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("切块参数必须是整数。")
    result = replace(cfg, chunk_size=size, overlap=overlap)
    validate(result)
    return result


def signature(cfg):
    if not cfg.paper.is_file():
        raise ValueError("找不到论文，请查看 papers 文件夹或 .env 的 PAPER_PATH。")
    return {"project": "paper-rag-v1", "sha256": hashlib.sha256(cfg.paper.read_bytes()).hexdigest(),
            "source": cfg.paper.name, "embedding_model": cfg.embedding_model,
            "dim": cfg.dim, "chunk_size": cfg.chunk_size, "overlap": cfg.overlap,
            "metric": "L2", "embedding_normalization": "l2"}


def collection_name(cfg):
    fingerprint = json.dumps(signature(cfg), sort_keys=True).encode()
    return "paper_" + hashlib.sha256(fingerprint).hexdigest()[:24]
