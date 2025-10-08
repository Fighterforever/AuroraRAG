#!/usr/bin/env python3
"""Export Neo4j graph to JSON for offline inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Neo4j graph to JSON")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    parser.add_argument(
        "--output", default="graph_export.json", help="Output path for exported graph"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from neo4j_storage import Neo4jKnowledgeGraph
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        print(
            "neo4j Python driver is not installed. Install dependencies via 'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        raise exc

    kg = Neo4jKnowledgeGraph(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
    )

    try:
        kg.export_graph(args.output)
        print(f"Exported graph to {args.output}")
    finally:
        kg.close()


if __name__ == "__main__":
    main()
