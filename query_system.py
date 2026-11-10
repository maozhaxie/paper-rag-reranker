"""Tutorial's query entry point, with a visible --search-only stage."""
import argparse
import json
from vector_retriever import answer_question, search_similar_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    result = ({"sources": search_similar_content(args.question, args.k)} if args.search_only
              else answer_question(args.question, args.k))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
