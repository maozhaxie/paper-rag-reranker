"""Real API evaluation, never run automatically. Page hit is a coarse retrieval metric."""
import argparse
from datetime import datetime
import json
from configuration import ROOT, settings, signature
from vector_retriever import search_similar_content, answer_question


def main():
    parser = argparse.ArgumentParser(description="会调用模型 API；--answers 还会生成答案。")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation/questions.json").read_text(encoding="utf-8"))
    cfg = settings()
    dataset = json.loads((ROOT / "evaluation/dataset_info.json").read_text(encoding="utf-8"))
    if signature(cfg)["sha256"] != dataset["paper_sha256"]:
        raise ValueError("当前论文不是测试题对应的版本；请先为新论文重写测试题和 dataset_info.json。")
    records = []
    hits = total = 0
    for case in cases:
        print(case["id"], case["question"], flush=True)
        result = (answer_question(case["question"], args.k, cfg) if args.answers else
                  {"sources": search_similar_content(case["question"], args.k, cfg)})
        pages = [s["page"] for s in result["sources"]]
        hit = bool(set(pages) & set(case["expected_pages"])) if case["answerable"] else None
        if hit is not None:
            total += 1
            hits += int(hit)
        records.append({**case, **result, "retrieved_pages": pages, "page_hit": hit,
                        "manual_answer_correct": None, "manual_citation_supported": None,
                        "manual_refusal_correct": None})
    report = {"signature": signature(cfg), "k": args.k, "answers_generated": args.answers,
              "page_hit_at_k": hits / total if total else None, "positive_cases": total,
              "note": "命中任一参考页的粗粒度指标，不是答案准确率，也不衡量多证据完整性；参考页未必穷尽。拒答与引用需人工核对。",
              "cases": records}
    output = ROOT / "outputs" / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
