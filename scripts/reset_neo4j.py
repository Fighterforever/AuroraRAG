#!/usr/bin/env python3
"""Reset Neo4j database with optional Docker invocation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout


def use_docker(args):
    container = args.container
    cmd = [
        "docker",
        "exec",
        "-i",
        container,
        "cypher-shell",
        "-u",
        args.user,
        "-p",
        args.password,
        "MATCH (n) DETACH DELETE n",
    ]
    print(f"Running command: {' '.join(cmd)}")
    run_command(cmd)

    if args.reset_constraints:
        constraint_cmd = [
            "docker",
            "exec",
            "-i",
            container,
            "cypher-shell",
            "-u",
            args.user,
            "-p",
            args.password,
            "CALL apoc.schema.assert({},{})",
        ]
        print(f"Running command: {' '.join(constraint_cmd)}")
        run_command(constraint_cmd)


def use_cypher_shell(args):
    if shutil.which("cypher-shell") is None:
        print("cypher-shell 不在当前 PATH 中。请安装 Neo4j 命令行工具，或改用 --use-docker 模式。", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "cypher-shell",
        "-a",
        args.uri,
        "-u",
        args.user,
        "-p",
        args.password,
        "MATCH (n) DETACH DELETE n",
    ]
    print(f"Running command: {' '.join(cmd)}")
    run_command(cmd)

    if args.reset_constraints:
        constraint_cmd = [
            "cypher-shell",
            "-a",
            args.uri,
            "-u",
            args.user,
            "-p",
            args.password,
            "CALL apoc.schema.assert({},{})",
        ]
        print(f"Running command: {' '.join(constraint_cmd)}")
        run_command(constraint_cmd)


def remove_local_state():
    to_remove = [
        Path("pipeline_state.json"),
        Path("extraction_progress.json"),
        Path("raw_extraction_results"),
    ]

    for path in to_remove:
        if path.is_file():
            print(f"Removing file: {path}")
            path.unlink()
        elif path.is_dir():
            print(f"Removing directory: {path}")
            for item in sorted(path.glob("**/*"), reverse=True):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            path.rmdir()


def main():
    parser = argparse.ArgumentParser(description="Reset Neo4j database and local pipeline state")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--container", default="aurorarag-neo4j", help="Docker container name")
    parser.add_argument("--use-docker", action="store_true", help="Use Docker exec to reset")
    parser.add_argument("--reset-constraints", action="store_true", help="Reset schema constraints via APOC")
    parser.add_argument("--clean-local", action="store_true", help="Remove local pipeline cache files")
    args = parser.parse_args()

    if args.use_docker:
        use_docker(args)
    else:
        use_cypher_shell(args)

    if args.clean_local:
        remove_local_state()

    print("Neo4j reset complete.")


if __name__ == "__main__":
    main()
