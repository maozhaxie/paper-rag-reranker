"""Flask API: preview is local; indexing/search/answer explicitly call services."""
import json
import threading
from flask import Blueprint, jsonify, request, send_file
from configuration import ROOT, settings, collection_name, preview_settings
from document_loader import load_document, save_report
from vector_retriever import validate_query


def create_blueprint():
    bp = Blueprint("vector", __name__, url_prefix="/api/vector")
    indexing = threading.Lock()

    @bp.get("/status")
    def status():
        cfg = settings()
        receipt = None
        try:
            receipt = json.loads((ROOT / "outputs/index_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return jsonify(success=True, paper=cfg.paper.name, paper_exists=cfg.paper.is_file(),
            key_configured=bool(cfg.key and cfg.key != "replace_with_your_key"),
            embedding_model=cfg.embedding_model, llm_model=cfg.llm_model,
            chunk_size=cfg.chunk_size, overlap=cfg.overlap,
            rerank_enabled=cfg.rerank_enabled, rerank_model=cfg.rerank_model,
            candidate_k=cfg.candidate_k,
            collection=collection_name(cfg) if cfg.paper.is_file() else "",
            last_index=receipt, database_checked=False)

    @bp.post("/preview")
    def preview():
        cfg = settings()
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("请求应为 JSON 对象。")
        cfg = preview_settings(cfg, data.get("chunk_size", cfg.chunk_size), data.get("overlap", cfg.overlap))
        report = load_document(cfg.paper, cfg.chunk_size, cfg.overlap)
        save_report(report, ROOT / "outputs")
        return jsonify(success=True, **report)

    @bp.get("/paper")
    def paper():
        cfg = settings()
        if not cfg.paper.is_file():
            raise ValueError("找不到论文。")
        return send_file(cfg.paper, mimetype="application/pdf", download_name=cfg.paper.name)

    @bp.post("/check-db")
    def check_db():
        from vector_db_manager import db_client
        db = db_client(settings())
        try:
            return jsonify(success=True, collections=db.list_collections())
        finally:
            db.close()

    @bp.post("/upload_document")
    def index():
        if not indexing.acquire(blocking=False):
            return jsonify(success=False, error="正在导入，请等本次导入结束。"), 409
        try:
            from vector_db_manager import process_file
            return jsonify(process_file(settings()))
        finally:
            indexing.release()

    def payload():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("请求应为 JSON 对象。")
        question, k = data.get("question"), data.get("k", 5)
        validate_query(question, k)
        rerank = data.get("rerank")
        if rerank is not None and not isinstance(rerank, bool):
            raise ValueError("重排开关必须是布尔值。")
        return question.strip(), k, rerank

    @bp.post("/search")
    def search():
        question, k, rerank = payload()
        if indexing.locked():
            return jsonify(success=False, error="导入中，请稍后查询。"), 409
        from vector_retriever import search_similar_content
        return jsonify(success=True, sources=search_similar_content(question, k, rerank=rerank))

    @bp.post("/query")
    def query():
        question, k, rerank = payload()
        if indexing.locked():
            return jsonify(success=False, error="导入中，请稍后查询。"), 409
        from vector_retriever import answer_question
        return jsonify(answer_question(question, k, rerank=rerank))

    return bp
