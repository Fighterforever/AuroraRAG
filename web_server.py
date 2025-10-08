"""基于 FastAPI 的服务包装。"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aurora_core import AuroraPipeline


logger = logging.getLogger("aurora_web")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: Dict[str, Any]


class IngestResponse(BaseModel):
    report: Dict[str, Any]


@lru_cache(maxsize=1)
def get_pipeline() -> AuroraPipeline:
    """延迟初始化管道，避免重复加载模型。"""
    return AuroraPipeline()


app = FastAPI(title="AuroraRAG", description="知识图谱增强问答服务")


@app.post("/ingest", response_model=IngestResponse)
async def run_ingest() -> IngestResponse:
    pipeline = get_pipeline()
    loop = asyncio.get_running_loop()
    try:
        report = await loop.run_in_executor(None, pipeline.ingest)
    except Exception as exc:  # pragma: no cover - 防御
        logger.exception("Ingest 执行失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(report=report)


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(None, pipeline.answer, request.question)
    except Exception as exc:  # pragma: no cover - 防御
        logger.exception("Query 执行失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(answer=answer)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    pipeline = get_pipeline()
    try:
        report = pipeline.health()
    except Exception as exc:  # pragma: no cover
        logger.exception("Health 检查失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return report
