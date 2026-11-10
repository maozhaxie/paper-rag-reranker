"""Indexing: text -> Bailian embeddings -> Milvus. Tutorial §6.1, completed."""
import hashlib
import json
import math
from configuration import ROOT, collection_name, signature
from document_loader import load_document, save_report


def model_client(cfg):
    if not cfg.key or cfg.key == "replace_with_your_key":
        raise ValueError("请在本地 .env 填写百炼 API Key，再使用入库和问答。读取论文不需要密钥。")
    from openai import OpenAI
    return OpenAI(api_key=cfg.key, base_url=cfg.base_url, timeout=60, max_retries=1)


def db_client(cfg):
    from pymilvus import MilvusClient
    return MilvusClient(uri=cfg.milvus_uri, timeout=20)


def normalize_vector(vector):
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("向量长度必须是有限的正数，无法归一化零向量或异常向量。")
    return [value / norm for value in vector]


def embed(client, cfg, texts):
    vectors = []
    # v1 permits 25 rows, v3/v4 10; keep batches at 10 for this small project.
    for offset in range(0, len(texts), 10):
        batch = texts[offset:offset + 10]
        result = client.embeddings.create(model=cfg.embedding_model, input=batch, encoding_format="float")
        items = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError("向量条数或返回索引与输入不一致。")
        for item in items:
            if len(item.embedding) != cfg.dim or not all(math.isfinite(x) for x in item.embedding):
                raise ValueError("向量维度或数值异常，请核对 EMBEDDING_DIM 和模型配置。")
            vectors.append(normalize_vector(item.embedding))
    return vectors


def ensure_collection(db, cfg, create=False):
    name = collection_name(cfg)
    if db.has_collection(collection_name=name):
        info = db.describe_collection(collection_name=name)
        try:
            existing = json.loads(info.get("description", ""))
        except (ValueError, TypeError):
            existing = None
        if existing != signature(cfg):
            raise ValueError("集合描述与当前论文/模型/切块参数不一致，拒绝混合查询。")
        db.load_collection(collection_name=name)
        return name
    if not create:
        raise ValueError("当前论文及切块配置尚未入库，请先点击‘将论文导入 Milvus’。")
    from pymilvus import MilvusClient, DataType
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                       description=json.dumps(signature(cfg)))
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=12000)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="page", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=cfg.dim)
    index = MilvusClient.prepare_index_params()
    index.add_index(field_name="embedding", index_type="FLAT", metric_type="L2", params={})
    db.create_collection(collection_name=name, schema=schema, index_params=index, consistency_level="Strong")
    db.load_collection(collection_name=name)
    return name


def process_file(cfg, progress=print):
    report = load_document(cfg.paper, cfg.chunk_size, cfg.overlap)
    save_report(report, ROOT / "outputs")
    client = model_client(cfg)
    db = None
    try:
        db = db_client(cfg)
        name = ensure_collection(db, cfg, create=True)
        for offset in range(0, len(report["chunks"]), 10):
            batch = report["chunks"][offset:offset + 10]
            vectors = embed(client, cfg, [row["text"] for row in batch])
            rows = [{"id": hashlib.sha256(f"{name}:{row['chunk_id']}".encode()).hexdigest(),
                     **row, "embedding": vector} for row, vector in zip(batch, vectors)]
            db.upsert(collection_name=name, data=rows)
            progress(f"已入库 {min(offset + 10, report['chunk_count'])}/{report['chunk_count']} 块")
        db.flush(collection_name=name)
        result = {"success": True, "file": cfg.paper.name, "chunks": report["chunk_count"],
                  "collection": name, "signature": signature(cfg)}
        path = ROOT / "outputs" / "index_state.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
        return result
    finally:
        if db is not None:
            db.close()
        client.close()


def check_receipt(cfg):
    path = ROOT / "outputs" / "index_state.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("尚无完整入库记录，请先导入论文；中断后可以重新导入。") from None
    if receipt.get("signature") != signature(cfg):
        raise ValueError("论文或向量/切块配置已经改变，请先重新导入当前版本。")
    return receipt
