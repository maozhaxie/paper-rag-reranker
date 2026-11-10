"""Tutorial's indexing entry point; --preview never calls a model."""
import argparse
import json
from configuration import ROOT, settings
from document_loader import load_document, save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    cfg = settings()
    if args.preview:
        result = load_document(cfg.paper, cfg.chunk_size, cfg.overlap)
        save_report(result, ROOT / "outputs")
        print(json.dumps({k: v for k, v in result.items() if k != "chunks"}, ensure_ascii=False, indent=2))
    else:
        from vector_db_manager import process_file
        print(json.dumps(process_file(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
