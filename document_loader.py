"""PDF -> page text -> LangChain Documents -> traceable chunks (tutorial §4.1)."""
from collections import Counter
import json
from pathlib import Path
import unicodedata


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))


def load_document(path, chunk_size=500, overlap=50):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader
    if not 0 <= overlap < chunk_size <= 1200:
        raise ValueError("切块参数需要满足 0 <= overlap < chunk_size <= 1200。")
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("请选择存在的 PDF 文件。")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("请先使用无密码的 PDF 副本。")
    pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    documents = [Document(page_content=text, metadata={"source": path.name, "page": i + 1})
                 for i, text in enumerate(pages) if text.strip()]
    if not documents:
        raise ValueError("未提取到文字。此项目不包含 OCR，请换用可选中文字的 PDF。")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""])
    chunks = splitter.split_documents(documents)
    for i, doc in enumerate(chunks, 1):
        doc.metadata["chunk_id"] = i
    rows = [{"text": doc.page_content, **doc.metadata} for doc in chunks]
    counts = Counter(row["page"] for row in rows)
    return {"file": path.name, "pages": len(pages), "characters": sum(map(len, pages)),
            "chunk_size": chunk_size, "overlap": overlap, "chunk_count": len(rows),
            "empty_pages": [i + 1 for i, text in enumerate(pages) if not text.strip()],
            "page_stats": [{"page": i + 1, "characters": len(text), "chunks": counts[i + 1]}
                           for i, text in enumerate(pages)], "chunks": rows}


def save_report(report, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chunks.json").write_text(json.dumps(report["chunks"], ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "document_report.json").write_text(json.dumps({k: v for k, v in report.items() if k != "chunks"},
        ensure_ascii=False, indent=2), encoding="utf-8")
