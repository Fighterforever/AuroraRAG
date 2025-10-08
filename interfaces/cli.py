"""Command-line interface for the AuroraRAG pipeline."""

from __future__ import annotations

import argparse
import json

from aurora_core import AuroraPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AuroraRAG Knowledge OS")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ingest", help="Run ingestion pipeline")

    qa_parser = subparsers.add_parser("qa", help="Ask a question against the knowledge graph")
    qa_parser.add_argument("question", nargs="?", default=None, help="Question to ask")

    subparsers.add_parser("full", help="Run ingestion then enter interactive QA mode")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    pipeline = AuroraPipeline()

    if args.command == "ingest":
        report = pipeline.ingest()
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.command == "qa":
        if args.question:
            result = pipeline.answer(args.question)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("进入多轮问答模式，输入 exit/quit 结束。")
            try:
                while True:
                    question = input("\n请输入问题: ")
                    if question.strip().lower() in {"exit", "quit"}:
                        break
                    if not question.strip():
                        continue
                    result = pipeline.answer(question)
                    print(json.dumps(result, ensure_ascii=False, indent=2))
            except KeyboardInterrupt:
                print("\n退出问答模式")

    elif args.command == "full":
        report = pipeline.ingest()
        print("Ingestion summary:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        try:
            while True:
                question = input("\n问题(输入exit退出): ")
                if question.strip().lower() in {"exit", "quit"}:
                    break
                result = pipeline.answer(question)
                print(json.dumps(result, ensure_ascii=False, indent=2))
        except KeyboardInterrupt:
            print("\n退出交互模式")


if __name__ == "__main__": 
    main()
