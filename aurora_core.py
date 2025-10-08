from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import yaml

from knowledge_graph_qa import KnowledgeGraphQA


# =========================
# 代理基础组件
# =========================


@dataclass
class AgentContext:
    """代理共享上下文，用于在多阶段流程间传递状态。"""

    run_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    scratchpad: Dict[str, Any] = field(default_factory=dict)

    def derive(self, **kwargs: Any) -> "AgentContext":
        new_metadata = {**self.metadata, **kwargs}
        return AgentContext(run_id=self.run_id, metadata=new_metadata, scratchpad=self.scratchpad)


class BaseAgent:
    """提供生命周期与日志能力的基类。"""

    def __init__(self, name: str, description: str, logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.description = description
        self.logger = logger or logging.getLogger(name)

    def before_run(self, context: AgentContext, **kwargs: Any) -> None:
        self.logger.debug("[%s] 上下文: %s", self.name, context.metadata)

    def after_run(self, context: AgentContext, result: Any) -> Any:
        if isinstance(result, dict):
            self.logger.debug("[%s] 输出键: %s", self.name, list(result.keys()))
        else:
            self.logger.debug("[%s] 输出: %s", self.name, result)
        return result

    def execute(self, context: AgentContext, **kwargs: Any) -> Any:  # pragma: no cover - 抽象方法
        raise NotImplementedError("代理需实现 execute() 方法")

    def run(self, context: AgentContext, **kwargs: Any) -> Any:
        self.before_run(context, **kwargs)
        result = self.execute(context, **kwargs)
        return self.after_run(context, result)


# -----------------
# 文档规划与抽取代理
# -----------------


@dataclass
class DocumentTask:
    """规划后的单文档处理任务。"""

    document_id: str
    source_path: str
    segments: List[Dict[str, Any]]


class StrategistAgent(BaseAgent):
    """负责将文档拆分为分段任务。"""

    def __init__(self, chunker, router) -> None:
        super().__init__(name="strategist", description="文档任务规划")
        self.chunker = chunker
        self.router = router

    def execute(self, context: AgentContext, **kwargs: Any) -> List[DocumentTask]:
        corpus = kwargs.get("corpus") or self.router.load_corpus()
        tasks: List[DocumentTask] = []

        for document in corpus:
            document_id = document.get("id") or str(uuid.uuid4())
            segments = self.chunker.chunk(document)
            tasks.append(DocumentTask(document_id=document_id, source_path=document["path"], segments=segments))
            self.logger.info("规划文档 %s 分段数量: %s", document["path"], len(segments))

        context.metadata["planned_tasks"] = len(tasks)
        return tasks


class EntityRelationExtractionAgent(BaseAgent):
    """对分段运行实体/关系抽取，支持多线程与速率控制。"""

    def __init__(
        self,
        qa_engine: KnowledgeGraphQA,
        schema_manager,
        max_workers: int = 4,
        rate_limit: float = 0.3,
    ) -> None:
        super().__init__(name="entity_relation_agent", description="抽取实体与关系")
        self.qa_engine = qa_engine
        self.schema_manager = schema_manager
        self.max_workers = max(1, max_workers)
        self.rate_limit = max(0.0, rate_limit)
        self._throttle_lock = threading.Lock()
        self._last_call = 0.0

    def execute(self, context: AgentContext, **kwargs: Any) -> List[Dict[str, Any]]:
        tasks: List[DocumentTask] = kwargs["tasks"]
        jobs: List[Tuple[int, DocumentTask, Dict[str, Any]]] = []
        for task in tasks:
            for segment in task.segments:
                if segment.get("text", "").strip():
                    jobs.append((len(jobs), task, segment))

        if not jobs:
            context.metadata["extracted_segments"] = 0
            return []

        result_map: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_segment, index, task, segment): index
                for index, task, segment in jobs
            }

            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    result_map[index] = result
                except Exception as exc: 
                    self.logger.error("分段 %s 抽取失败: %s", index, exc)

        results = [result_map[idx] for idx in sorted(result_map.keys())]
        context.metadata["extracted_segments"] = len(results)
        return results

    def _throttle(self) -> None:
        if self.rate_limit <= 0:
            return
        with self._throttle_lock:
            elapsed = time.time() - self._last_call
            wait = self.rate_limit - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _process_segment(self, index: int, task: DocumentTask, segment: Dict[str, Any]) -> Dict[str, Any]:
        text = segment.get("text", "")
        self._throttle()
        start = time.time()
        try:
            entities, relationships = self.qa_engine.extract_entities_and_relationships(text)
        except Exception as exc: 
            self.logger.error("LLM 抽取失败 segment=%s: %s", segment.get("id"), exc)
            entities, relationships = [], []
        latency = time.time() - start

        return {
            "document_id": task.document_id,
            "source_path": task.source_path,
            "segment_id": segment["id"],
            "entities": entities,
            "relationships": relationships,
            "events": [],
            "attributes": [],
            "metadata": {
                "latency": latency,
                "schema_version": self.schema_manager.current_version,
            },
        }


class AttributeExtractionAgent(BaseAgent):
    """利用 LLM 抽取描述性属性。"""

    def __init__(self, qa_engine: KnowledgeGraphQA) -> None:
        super().__init__(name="attribute_agent", description="抽取属性信息")
        self.qa_engine = qa_engine

    def execute(self, context: AgentContext, **kwargs: Any) -> List[Dict[str, Any]]:
        tasks: List[DocumentTask] = kwargs.get("tasks", [])
        records: List[Dict[str, Any]] = []

        for task in tasks:
            for segment in task.segments:
                text = segment.get("text", "")
                if not text.strip():
                    continue
                extracted = self.qa_engine.extract_attributes(text)
                for attr in extracted:
                    entity_name = attr.get("entity")
                    attribute_name = attr.get("attribute")
                    value = attr.get("value")
                    if not (entity_name and attribute_name and value):
                        continue
                    records.append(
                        {
                            "document_id": task.document_id,
                            "segment_id": segment["id"],
                            "entity_candidate": entity_name,
                            "attribute": attribute_name,
                            "value": value,
                            "unit": attr.get("unit", ""),
                            "context": attr.get("evidence", text[:200]),
                            "attribute_type": attr.get("attribute_type", "ATTRIBUTE"),
                        }
                    )

        context.metadata["extracted_attributes"] = len(records)
        return records


class EventExtractionAgent(BaseAgent):
    """利用 LLM 抽取事件/时间轴信息。"""

    def __init__(self, qa_engine: KnowledgeGraphQA) -> None:
        super().__init__(name="event_agent", description="抽取事件信息")
        self.qa_engine = qa_engine

    def execute(self, context: AgentContext, **kwargs: Any) -> List[Dict[str, Any]]:
        tasks: List[DocumentTask] = kwargs.get("tasks", [])
        events: List[Dict[str, Any]] = []

        for task in tasks:
            for segment in task.segments:
                text = segment.get("text", "")
                if not text.strip():
                    continue
                extracted = self.qa_engine.extract_events(text)
                for idx, event in enumerate(extracted):
                    label = event.get("label")
                    if not label:
                        continue
                    event_id = event.get("id") or f"event_{task.document_id}_{segment['id']}_{idx}"
                    events.append(
                        {
                            "id": event_id,
                            "type": event.get("type", "EVENT"),
                            "label": label,
                            "participants": event.get("participants", []),
                            "document_id": task.document_id,
                            "segment_id": segment["id"],
                            "context": event.get("context", text[:200]),
                            "timestamp": event.get("timestamp"),
                        }
                    )

        context.metadata["extracted_events"] = len(events)
        return events


class ConsistencyCritic(BaseAgent):
    """检测多段抽取结果中的重复实体与关系。"""

    def __init__(self) -> None:
        super().__init__(name="consistency_critic", description="重复检测")

    def execute(self, context: AgentContext, **kwargs: Any) -> Dict[str, Any]:
        extractions: List[Dict[str, Any]] = kwargs["extractions"]
        entity_occurrences = defaultdict(list)
        relationship_occurrences = defaultdict(list)

        for extraction in extractions:
            for entity in extraction.get("entities", []):
                key = (entity.get("name"), entity.get("type"))
                entity_occurrences[key].append(extraction["segment_id"])

            for rel in extraction.get("relationships", []):
                key = (rel.get("source"), rel.get("target"), rel.get("relationship"))
                relationship_occurrences[key].append(extraction["segment_id"])

        duplicate_entities = [
            {
                "key": f"{name}::{etype}",
                "segments": sorted(set(segments)),
            }
            for (name, etype), segments in entity_occurrences.items()
            if len(set(segments)) > 1
        ]
        duplicate_relationships = [
            {
                "key": f"{src}->{rel}->{tgt}",
                "segments": sorted(set(segments)),
            }
            for (src, tgt, rel), segments in relationship_occurrences.items()
            if len(set(segments)) > 1
        ]

        flagged_segments = set()
        for item in duplicate_entities:
            flagged_segments.update(item.get("segments", []))
        for item in duplicate_relationships:
            flagged_segments.update(item.get("segments", []))

        context.metadata["flagged_segments"] = len(flagged_segments)
        return {
            "duplicates": {
                "entities": duplicate_entities,
                "relationships": duplicate_relationships,
            },
            "flagged_segments": list(flagged_segments),
        }


@dataclass
class ProvenanceRecord:
    document_id: str
    segment_id: str
    source_path: str
    extractor: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class ChangeRecord:
    change_type: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Optional[ProvenanceRecord] = None


@dataclass
class ChangeSet:
    run_id: str
    records: List[ChangeRecord]
    critic_annotation: Dict[str, Any] = field(default_factory=dict)

    def entity_count(self) -> int:
        return sum(len(record.entities) for record in self.records)

    def relationship_count(self) -> int:
        return sum(len(record.relationships) for record in self.records)


class KnowledgeCuratorAgent(BaseAgent):
    """将抽取结果转换为变更集供 KnowledgeOS 应用。"""

    def __init__(self) -> None:
        super().__init__(name="knowledge_curator", description="构建知识变更集")

    def execute(self, context: AgentContext, **kwargs: Any) -> ChangeSet:
        extractions: List[Dict[str, Any]] = kwargs["extractions"]
        critic_report: Dict[str, Any] = kwargs.get("critic_report", {})

        change_records: List[ChangeRecord] = []
        for extraction in extractions:
            provenance = ProvenanceRecord(
                document_id=extraction["document_id"],
                segment_id=extraction["segment_id"],
                source_path=extraction["source_path"],
                extractor=self.name,
                metadata={
                    "latency": extraction.get("metadata", {}).get("latency"),
                    "schema_version": extraction.get("metadata", {}).get("schema_version"),
                },
            )

            change_records.append(
                ChangeRecord(
                    change_type="merge",
                    entities=extraction.get("entities", []),
                    relationships=extraction.get("relationships", []),
                    events=extraction.get("events", []),
                    attributes=extraction.get("attributes", []),
                    provenance=provenance,
                )
            )

        context.metadata["change_records"] = len(change_records)
        return ChangeSet(run_id=context.run_id, records=change_records, critic_annotation=critic_report)


class AggregatorAgent(BaseAgent):
    """合并分段抽取结果，执行别名归一化与关系去重。"""

    def __init__(
        self,
        qa_engine: Optional[KnowledgeGraphQA] = None,
        alias_dictionary: Optional[Dict[str, List[str]]] = None,
        alias_dict_path: Optional[str] = None,
        max_workers: int = 4,
    ) -> None:
        super().__init__(name="aggregator", description="归并实体与关系")
        self.qa_engine = qa_engine
        self.alias_dictionary = {k: list(v or []) for k, v in (alias_dictionary or {}).items()}
        self.alias_dict_path = Path(alias_dict_path).resolve() if alias_dict_path else None
        self.max_workers = max(1, max_workers)
        self._relation_label_cache: Dict[Tuple[str, str, str], str] = {}
        self._relation_equivalence_cache: Dict[Tuple[str, str, str, str], bool] = {}
        self._relation_cache_lock = threading.Lock()

    def execute(self, context: AgentContext, **kwargs: Any) -> Dict[str, Any]:
        extractions: List[Dict[str, Any]] = kwargs.get("extractions", [])
        attributes: List[Dict[str, Any]] = kwargs.get("attributes", [])
        events: List[Dict[str, Any]] = kwargs.get("events", [])

        canonical_entities: Dict[str, Dict[str, Any]] = {}
        alias_lookup: Dict[str, str] = {}

        for name, aliases in self.alias_dictionary.items():
            for variant in aliases + [name]:
                alias_lookup[variant] = name
                alias_lookup[self._normalize_name(variant)] = name

        entity_duplicates: defaultdict[Tuple[str, str], List[str]] = defaultdict(list)

        for extraction in extractions:
            segment_id = extraction.get("segment_id")
            for entity in extraction.get("entities", []):
                original_name = (entity.get("name") or "").strip()
                if not original_name:
                    continue

                mapped = alias_lookup.get(original_name) or alias_lookup.get(self._normalize_name(original_name))
                name = mapped or original_name
                variants = self._alias_variants(entity)
                norm = self._normalize_name(name)
                canonical = canonical_entities.get(norm)
                if canonical is None:
                    canonical_id = entity.get("id") or self._make_entity_id(norm)
                    canonical = {
                        "id": canonical_id,
                        "name": name,
                        "type": entity.get("type", "CONCEPT") or "CONCEPT",
                        "description": entity.get("description") or f"{name} 实体",
                        "aliases": set(variants) or {name},
                    }
                    canonical_entities[norm] = canonical
                else:
                    if segment_id:
                        entity_duplicates[(canonical["name"], canonical["type"])].append(segment_id)
                    canonical["aliases"].update(variants)
                    description = entity.get("description")
                    if description and len(description) > len(canonical.get("description", "")):
                        canonical["description"] = description

                for alias in variants:
                    alias_lookup[alias] = canonical["name"]
                    alias_lookup[self._normalize_name(alias)] = canonical["name"]

        raw_relationships: List[Dict[str, Any]] = []
        for extraction in extractions:
            segment_id = extraction.get("segment_id")
            for relationship in extraction.get("relationships", []):
                source = self._resolve_alias(alias_lookup, relationship.get("source"))
                target = self._resolve_alias(alias_lookup, relationship.get("target"))
                rel_type = relationship.get("relationship", "RELATED_TO")

                if not source or not target:
                    continue

                source_norm = self._normalize_name(source)
                target_norm = self._normalize_name(target)

                source_entity = canonical_entities.get(source_norm)
                target_entity = canonical_entities.get(target_norm)
                if not source_entity or not target_entity:
                    continue

                raw_relationships.append(
                    {
                        "id": relationship.get("id"),
                        "raw_type": rel_type,
                        "description": relationship.get("description", ""),
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "segment_id": segment_id,
                    }
                )

        aggregated_relationships, relationship_duplicates = self._aggregate_relationships(raw_relationships)

        aggregated_events = [
            {
                **event,
                "participants": [self._resolve_alias(alias_lookup, p) for p in event.get("participants", [])],
            }
            for event in events
        ]

        aggregated_attributes = [
            {
                **attribute,
                "entity_candidate": alias_lookup.get(attribute.get("entity_candidate"))
                or alias_lookup.get(self._normalize_name(attribute.get("entity_candidate", "")))
                or attribute.get("entity_candidate"),
            }
            for attribute in attributes
        ]

        aggregated_entities: List[Dict[str, Any]] = []
        for entity in canonical_entities.values():
            aliases = sorted(entity.get("aliases", []))
            entity["aliases"] = aliases
            aggregated_entities.append(entity)

        # 使用 LLM 进一步合并语义相似的实体
        if self.qa_engine and hasattr(self.qa_engine, "confirm_entity_equivalence"):
            aggregated_entities = self._merge_similar_entities_with_llm(aggregated_entities)

        self._persist_aliases(aggregated_entities)

        aggregated_entry = {
            "document_id": "__aggregated__",
            "segment_id": "__aggregated__",
            "source_path": "",
            "entities": aggregated_entities,
            "relationships": aggregated_relationships,
            "events": aggregated_events,
            "attributes": aggregated_attributes,
            "metadata": {
                "merged_segments": len(extractions),
            },
        }

        return {
            "aggregated_extractions": [aggregated_entry],
            "events": aggregated_events,
            "attributes": aggregated_attributes,
            "duplicates": relationship_duplicates,
        }

    # --- 内部工具方法 ---

    def _aggregate_relationships(
        self, raw_relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not raw_relationships:
            return [], []

        normalized = self._normalize_relationships(raw_relationships)
        aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        segment_map: defaultdict[Tuple[str, str, str], set[str]] = defaultdict(set)
        relation_index: defaultdict[Tuple[str, str], List[str]] = defaultdict(list)

        for relation in normalized:
            canonical_label = relation.get("canonical_label") or relation.get("raw_type") or "RELATED_TO"
            source_entity = relation["source_entity"]
            target_entity = relation["target_entity"]
            source_name = source_entity["name"]
            target_name = target_entity["name"]
            canonical_label = self._resolve_equivalent_label(
                source_name,
                target_name,
                canonical_label,
                relation,
                aggregated,
                relation_index[(source_name, target_name)],
            )
            key = (source_name, canonical_label, target_name)

            existing = aggregated.get(key)
            relation_id = self._make_relationship_id(
                source_entity["id"], canonical_label, target_entity["id"]
            )
            description = relation.get("description", "")

            if existing is None:
                aggregated[key] = {
                    "id": relation_id,
                    "relationship": canonical_label,
                    "description": description,
                    "source": source_name,
                    "target": target_name,
                    "source_id": source_entity["id"],
                    "target_id": target_entity["id"],
                }
            else:
                existing["id"] = relation_id
                if description and len(description) > len(existing.get("description", "")):
                    existing["description"] = description

            segment_id = relation.get("segment_id")
            if segment_id:
                segment_map[key].add(segment_id)

            if canonical_label not in relation_index[(source_name, target_name)]:
                relation_index[(source_name, target_name)].append(canonical_label)

        duplicates = [
            {
                "key": f"{src}->{rel_type}->{tgt}",
                "segments": sorted(seg_ids),
            }
            for (src, rel_type, tgt), seg_ids in segment_map.items()
            if len(seg_ids) > 1
        ]

        return list(aggregated.values()), duplicates

    def _normalize_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not relationships:
            return []

        if not self.qa_engine or not hasattr(self.qa_engine, "normalize_relationship_label"):
            for relation in relationships:
                relation["canonical_label"] = relation.get("raw_type") or "RELATED_TO"
            return relationships

        def worker(relation: Dict[str, Any]) -> Dict[str, Any]:
            relation["canonical_label"] = self._normalize_relationship_label(
                relation["source_entity"]["name"],
                relation["target_entity"]["name"],
                relation.get("raw_type"),
                relation.get("description", ""),
            )
            return relation

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(worker, relationships))

    def _normalize_relationship_label(
        self, source: str, target: str, raw_label: Optional[str], description: str
    ) -> str:
        label = (raw_label or "RELATED_TO").strip() or "RELATED_TO"
        cache_key = (source.lower(), label.lower(), target.lower())

        with self._relation_cache_lock:
            cached = self._relation_label_cache.get(cache_key)
        if cached:
            return cached

        canonical = label
        if self.qa_engine and hasattr(self.qa_engine, "normalize_relationship_label"):
            try:
                response = self.qa_engine.normalize_relationship_label(source, target, label, description)
                if isinstance(response, dict):
                    canonical = response.get("canonical") or canonical
                elif isinstance(response, str):
                    canonical = response or canonical
            except Exception as exc:  # pragma: no cover - 防御
                self.logger.warning("关系归一化失败: %s", exc)

        canonical = canonical or label or "RELATED_TO"
        with self._relation_cache_lock:
            self._relation_label_cache[cache_key] = canonical
        return canonical

    def _resolve_equivalent_label(
        self,
        source_name: str,
        target_name: str,
        candidate_label: str,
        relation: Dict[str, Any],
        aggregated: Dict[Tuple[str, str, str], Dict[str, Any]],
        existing_labels: List[str],
    ) -> str:
        normalized_candidate = self._normalize_label_key(candidate_label)
        for existing_label in existing_labels:
            if self._normalize_label_key(existing_label) == normalized_candidate:
                return existing_label

            if self._relationships_equivalent(
                source_name,
                target_name,
                existing_label,
                candidate_label,
                relation,
                aggregated,
            ):
                return existing_label

        return candidate_label

    def _relationships_equivalent(
        self,
        source_name: str,
        target_name: str,
        label_a: str,
        label_b: str,
        relation: Dict[str, Any],
        aggregated: Dict[Tuple[str, str, str], Dict[str, Any]],
    ) -> bool:
        norm_a = self._normalize_label_key(label_a)
        norm_b = self._normalize_label_key(label_b)
        if norm_a == norm_b:
            return True

        sorted_labels = tuple(sorted([norm_a, norm_b]))
        cache_key = (source_name.lower(), target_name.lower(), sorted_labels[0], sorted_labels[1])
        with self._relation_cache_lock:
            if cache_key in self._relation_equivalence_cache:
                return self._relation_equivalence_cache[cache_key]

        same = False
        if self.qa_engine and hasattr(self.qa_engine, "compare_relationship_labels"):
            try:
                description_b = relation.get("description", "")
                existing = aggregated.get((source_name, label_a, target_name), {})
                description_a = existing.get("description", "")
                response = self.qa_engine.compare_relationship_labels(
                    source_name,
                    target_name,
                    label_a,
                    label_b,
                    context_a=description_a,
                    context_b=description_b,
                )
                same = bool(response.get("same"))
            except Exception:  # pragma: no cover - 防御
                same = False

        with self._relation_cache_lock:
            self._relation_equivalence_cache[cache_key] = same
        return same

    def _normalize_relationships_cache_reset(self) -> None:
        with self._relation_cache_lock:
            self._relation_label_cache.clear()
            self._relation_equivalence_cache.clear()

    def _alias_variants(self, entity: Dict[str, Any]) -> List[str]:
        variants: set[str] = set()
        primary = entity.get("name", "")
        if primary:
            variants.add(primary.strip())
        for alias in entity.get("aliases", []) or []:
            if alias:
                variants.add(alias.strip())

        expanded: set[str] = set()
        for alias in variants:
            if not alias:
                continue
            expanded.add(alias)
        return [a for a in expanded if a]

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "")

    def _make_entity_id(self, norm_name: str) -> str:
        return f"entity::{norm_name}"

    def _make_relationship_id(self, source_id: str, rel_type: str, target_id: str) -> str:
        clean_rel = re.sub(r"[^a-z0-9]+", "-", rel_type.lower())
        return f"rel::{source_id}::{clean_rel}::{target_id}"

    def _resolve_alias(self, lookup: Dict[str, str], name: Optional[str]) -> str:
        if not name:
            return ""
        norm = self._normalize_name(name)
        return lookup.get(name) or lookup.get(norm) or name

    def _persist_aliases(self, aggregated_entities: List[Dict[str, Any]]) -> None:
        if not self.alias_dict_path:
            return

        updated = False
        for entity in aggregated_entities:
            canonical = entity.get("name")
            if not canonical:
                continue
            alias_set = set(entity.get("aliases", [])) | {canonical}
            stored = set(self.alias_dictionary.get(canonical, [])) | {canonical}
            if alias_set != stored:
                self.alias_dictionary[canonical] = sorted(alias_set)
                updated = True

        if not updated:
            return

        self.alias_dict_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alias_dict_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.alias_dictionary, handle, allow_unicode=True, sort_keys=True)

    def _normalize_label_key(self, label: str) -> str:
        return re.sub(r"\s+", "", (label or "").lower())

    def _merge_similar_entities_with_llm(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 LLM 合并语义相似的实体"""
        if len(entities) <= 1:
            return entities

        # 按类型分组，只比较同类型的实体
        type_groups: Dict[str, List[int]] = defaultdict(list)
        for i, entity in enumerate(entities):
            entity_type = entity.get("type", "CONCEPT")
            type_groups[entity_type].append(i)

        # 对实体两两比较，使用并发加速
        def check_similarity(pair: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            i, j = pair
            entity_a = entities[i]
            entity_b = entities[j]
            
            # 跳过名称完全相同的实体（已经合并过）
            if self._normalize_name(entity_a["name"]) == self._normalize_name(entity_b["name"]):
                return None
            
            # 预筛选：名称长度差异过大，跳过
            name_a = entity_a["name"]
            name_b = entity_b["name"]
            if abs(len(name_a) - len(name_b)) > max(len(name_a), len(name_b)) * 0.7:
                return None
            
            # 使用 LLM 判断是否为同一实体
            try:
                context = f"{entity_a.get('description', '')} | {entity_b.get('description', '')}"
                response = self.qa_engine.confirm_entity_equivalence(
                    name_a,
                    name_b,
                    context
                )
                if response.get("same", False):
                    return (i, j)
            except Exception:
                pass
            return None

        # 只对同类型的实体生成比较对，减少比较次数
        pairs: List[Tuple[int, int]] = []
        for entity_type, indices in type_groups.items():
            if len(indices) > 1:
                pairs.extend([(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))])
        
        # 并发执行相似度检查，优化批处理
        similar_pairs: List[Tuple[int, int]] = []
        batch_size = min(50, max(10, len(pairs) // self.max_workers))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 使用 as_completed 提高响应性
            futures = [executor.submit(check_similarity, pair) for pair in pairs]
            for future in as_completed(futures):
                try:
                    result_pair = future.result()
                    if result_pair:
                        similar_pairs.append(result_pair)
                except Exception:
                    pass

        # 构建合并映射（使用并查集思想）
        parent: Dict[int, int] = {i: i for i in range(len(entities))}
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        for i, j in similar_pairs:
            union(i, j)

        # 按组合并实体
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(entities)):
            groups[find(i)].append(i)

        # 合并每个组的实体
        merged_entities: List[Dict[str, Any]] = []
        for root, group in groups.items():
            if len(group) == 1:
                merged_entities.append(entities[group[0]])
            else:
                # 合并多个实体：使用第一个作为主实体，合并其他信息
                primary = entities[group[0]].copy()
                all_aliases = set(primary.get("aliases", []))
                longest_description = primary.get("description", "")
                
                for idx in group[1:]:
                    entity = entities[idx]
                    all_aliases.update(entity.get("aliases", []))
                    all_aliases.add(entity["name"])
                    desc = entity.get("description", "")
                    if len(desc) > len(longest_description):
                        longest_description = desc
                
                primary["aliases"] = sorted(all_aliases)
                primary["description"] = longest_description
                merged_entities.append(primary)

        return merged_entities


# -----------------
# 文档加载与切分
# -----------------


class DocumentStore:
    """基于文件夹的简单文档存储。"""

    def __init__(self, config_path: str = "settings.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _target_directory(self) -> Path:
        paths_cfg = self.config.get("paths", {})
        return Path(paths_cfg.get("target_directory", "sample_data"))

    def list_documents(self) -> List[Path]:
        target_dir = self._target_directory()
        if not target_dir.exists():
            return []
        return [path for path in target_dir.glob("**/*") if path.is_file() and path.suffix.lower() in {".md", ".txt"}]

    def load(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for path in self.list_documents():
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            documents.append(
                {
                    "id": path.stem,
                    "path": str(path),
                    "text": text,
                }
            )
        return documents


class HierarchicalChunker:
    """基于标题/段落的层次化分段器。"""

    def __init__(
        self,
        max_chars: int = 2000,
        min_chars: int = 800,
        overlap_chars: int = 200,
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = (document.get("text", "") or "").replace("\r\n", "\n")

        sections = self._split_sections(text)
        raw_chunks: List[str] = []

        for _, section_body in sections:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_body) if p.strip()]
            if not paragraphs:
                continue

            current = ""
            for paragraph in paragraphs:
                candidate = f"{current}\n\n{paragraph}" if current else paragraph
                if len(candidate) <= self.max_chars:
                    current = candidate
                else:
                    if current:
                        raw_chunks.append(current.strip())
                        current = self._apply_overlap(current, paragraph)
                    current = f"{current}\n\n{paragraph}" if current else paragraph

            if current:
                raw_chunks.append(current.strip())

        merged_chunks = self._post_process(raw_chunks)
        segments: List[Dict[str, Any]] = []
        for index, chunk_text in enumerate(merged_chunks):
            segments.append(
                {
                    "id": f"{document.get('id')}_{index}",
                    "text": chunk_text,
                    "document_id": document.get("id"),
                    "source_path": document.get("path", ""),
                }
            )
        return segments

    def _split_sections(self, text: str) -> List[Tuple[str, str]]:
        sections: List[Tuple[str, str]] = []
        current_title = ""
        buffer: List[str] = []
        heading_pattern = re.compile(r"^(#+)\s+(.*)")
        for line in text.splitlines():
            match = heading_pattern.match(line)
            if match:
                if buffer:
                    sections.append((current_title, "\n".join(buffer).strip()))
                    buffer = []
                current_title = match.group(2).strip()
            else:
                buffer.append(line)
        if buffer:
            sections.append((current_title, "\n".join(buffer).strip()))
        if not sections:
            sections.append(("", text))
        return sections

    def _apply_overlap(self, previous_chunk: str, next_paragraph: str) -> str:
        if self.overlap_chars <= 0:
            return ""
        overlap = previous_chunk[-self.overlap_chars :]
        overlap = overlap.split("\n\n", 1)[-1] if "\n\n" in overlap else overlap
        combined = (overlap + "\n\n" + next_paragraph).strip()
        return combined[: self.overlap_chars]

    def _post_process(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []
        merged: List[str] = []
        buffer = chunks[0]
        for chunk in chunks[1:]:
            if len(buffer) < self.min_chars:
                buffer = f"{buffer}\n\n{chunk}"
                if len(buffer) <= self.max_chars:
                    continue
            merged.append(buffer.strip())
            buffer = chunk
        if buffer:
            if merged and len(buffer) < self.min_chars:
                merged[-1] = f"{merged[-1]}\n\n{buffer}".strip()
            else:
                merged.append(buffer.strip())
        return [chunk.strip() for chunk in merged if chunk.strip()]


class DocumentRouter:
    """简单的文档路由器，当前直接读取文档存储。"""

    def __init__(self, store: Optional[DocumentStore] = None) -> None:
        self.store = store or DocumentStore()

    def load_corpus(self) -> List[Dict[str, Any]]:
        return self.store.load()


# -----------------
# 图谱与模式管理
# -----------------


@dataclass
class SchemaSnapshot:
    version: int
    entity_types: Dict[str, Dict[str, Any]]
    relationship_types: Dict[str, Dict[str, Any]]
    event_types: Dict[str, Dict[str, Any]]
    attribute_types: Dict[str, Dict[str, Any]]


class SchemaManager:
    """跟踪实体/关系/事件/属性的模式统计。"""

    def __init__(self) -> None:
        self.current_version = 1
        self.entity_types: Dict[str, Dict[str, Any]] = {}
        self.relationship_types: Dict[str, Dict[str, Any]] = {}
        self.event_types: Dict[str, Dict[str, Any]] = {}
        self.attribute_types: Dict[str, Dict[str, Any]] = {}
        self.history: List[SchemaSnapshot] = []

    def register_entities(self, entities: List[Dict[str, Any]]) -> None:
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            name = entity.get("name")
            if not name:
                continue
            entry = self.entity_types.setdefault(entity_type, {"examples": set(), "count": 0})
            entry["examples"].add(name)
            entry["count"] += 1

    def register_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        for relationship in relationships:
            relationship_type = relationship.get("relationship", "RELATED_TO")
            key = f"{relationship_type}|{relationship.get('source')}->{relationship.get('target')}"
            entry = self.relationship_types.setdefault(relationship_type, {"examples": set(), "count": 0})
            entry["examples"].add(key)
            entry["count"] += 1

    def register_events(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            event_type = event.get("type", "EVENT")
            label = event.get("label") or event.get("description") or "event"
            entry = self.event_types.setdefault(event_type, {"examples": set(), "count": 0})
            entry["examples"].add(label)
            entry["count"] += 1

    def register_attributes(self, attributes: List[Dict[str, Any]]) -> None:
        for attribute in attributes:
            attr_type = attribute.get("attribute_type", "ATTRIBUTE")
            label = attribute.get("attribute") or attribute.get("name") or "attribute"
            entry = self.attribute_types.setdefault(attr_type, {"examples": set(), "count": 0})
            entry["examples"].add(label)
            entry["count"] += 1

    def snapshot(self) -> SchemaSnapshot:
        snapshot = SchemaSnapshot(
            version=self.current_version,
            entity_types={k: {"examples": sorted(v["examples"]), "count": v["count"]} for k, v in self.entity_types.items()},
            relationship_types={k: {"examples": sorted(v["examples"]), "count": v["count"]} for k, v in self.relationship_types.items()},
            event_types={k: {"examples": sorted(v["examples"]), "count": v["count"]} for k, v in self.event_types.items()},
            attribute_types={k: {"examples": sorted(v["examples"]), "count": v["count"]} for k, v in self.attribute_types.items()},
        )
        self.history.append(snapshot)
        self.current_version += 1
        return snapshot


@dataclass
class KnowledgeSnapshot:
    version: int
    entity_count: int
    relationship_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeOS:
    """管理知识图谱版本、模式与社区画像的核心组件。"""

    def __init__(
        self,
        qa_engine: KnowledgeGraphQA,
        schema_manager: Optional[SchemaManager] = None,
        neo4j_adapter: Optional["Neo4jAdapter"] = None,
    ) -> None:
        if neo4j_adapter is None:
            raise ValueError("KnowledgeOS 需要 Neo4jAdapter 提供存储能力")

        self.qa_engine = qa_engine
        self.schema_manager = schema_manager or SchemaManager()
        self.neo4j_adapter = neo4j_adapter
        self.logger = logging.getLogger(self.__class__.__name__)

        self.version = 0
        self.snapshots: List["KnowledgeSnapshot"] = []
        self.entity_index: Dict[str, Dict[str, Any]] = {}
        self.relationship_index: Dict[str, Dict[str, Any]] = {}
        self.provenance_log: List[ProvenanceRecord] = []
        self.events: List[Dict[str, Any]] = []
        self.attributes: List[Dict[str, Any]] = []
        self.current_communities: Dict[str, Dict[str, Any]] = {}
        self.alias_index: Dict[str, str] = {}

    def apply_change_set(self, change_set: ChangeSet) -> "KnowledgeSnapshot":
        entity_updates: List[Dict[str, Any]] = []
        relationship_updates: List[Dict[str, Any]] = []
        event_updates: List[Dict[str, Any]] = []
        attribute_updates: List[Dict[str, Any]] = []

        existing_entities = {entity["name"]: entity["id"] for entity in self.get_entities(refresh=True)}

        for record in change_set.records:
            for entity in record.entities:
                key = entity.get("name")
                if not key:
                    continue
                if existing_entities.get(key):
                    entity_id = existing_entities[key]
                else:
                    entity_id = entity.get("id") or key
                stored = {**entity, "id": entity_id}
                existing_entities[key] = entity_id
                existing_entities[entity_id] = entity_id
                entity_updates.append(stored)
                self.schema_manager.register_entities([stored])
                for alias in [stored.get("name", "")] + list(stored.get("aliases", [])):
                    if alias:
                        self.alias_index[alias.strip().lower()] = entity_id

            for relationship in record.relationships:
                relationship_type = relationship.get("relationship", "RELATED_TO")
                source_name = relationship.get("source") or relationship.get("source_name")
                target_name = relationship.get("target") or relationship.get("target_name")
                if not source_name or not target_name:
                    continue
                source_id = relationship.get("source_id") or existing_entities.get(source_name, source_name)
                target_id = relationship.get("target_id") or existing_entities.get(target_name, target_name)
                rel_id = relationship.get("id") or f"{source_id}::{relationship_type}::{target_id}"
                stored_rel = {
                    "id": rel_id,
                    "relationship": relationship_type,
                    "description": relationship.get("description", ""),
                    "source_id": source_id,
                    "target_id": target_id,
                    "source": source_name,
                    "target": target_name,
                }
                relationship_updates.append(stored_rel)
                self.schema_manager.register_relationships([
                    {
                        "relationship": relationship_type,
                        "source": source_name,
                        "target": target_name,
                        "description": relationship.get("description", ""),
                    }
                ])

            if record.provenance:
                self.provenance_log.append(record.provenance)

            if record.events:
                self.schema_manager.register_events(record.events)
                event_updates.extend(record.events)

            if record.attributes:
                self.schema_manager.register_attributes(record.attributes)
                attribute_updates.extend(record.attributes)

        if entity_updates:
            self.neo4j_adapter.upsert_entities(entity_updates)

        relationship_payload = [
            {
                "id": rel["id"],
                "relationship": rel["relationship"],
                "description": rel["description"],
                "source": rel["source_id"],
                "target": rel["target_id"],
            }
            for rel in relationship_updates
        ]
        if relationship_payload:
            self.neo4j_adapter.upsert_relationships(relationship_payload)

        entities = self.get_entities(refresh=True)
        relationships = self.get_relationships(refresh=True)

        if event_updates:
            self.events.extend(event_updates)
        if attribute_updates:
            self.attributes.extend(attribute_updates)

        if entities or relationships:
            qa_relationships = [
                {
                    "source": rel.get("source"),
                    "target": rel.get("target"),
                    "relationship": rel.get("relationship"),
                    "description": rel.get("description"),
                }
                for rel in relationships
            ]
            self.qa_engine.build_knowledge_graph(
                entities,
                qa_relationships,
                sync_to_neo4j=False,
            )
            communities = self.qa_engine.detect_communities()
            community_profiles = self._build_community_profiles(communities, entities, relationships)
            try:
                self.neo4j_adapter.replace_communities(community_profiles)
            except Exception as exc:  # pragma: no cover - 防御
                self.logger.warning("社区持久化失败: %s", exc)
        else:
            community_profiles = {}

        self.current_communities = community_profiles or {}

        self.version += 1
        snapshot = KnowledgeSnapshot(
            version=self.version,
            entity_count=len(entities),
            relationship_count=len(relationships),
            metadata={
                "critic_annotation": change_set.critic_annotation,
                "communities": len(self.current_communities),
                "events": len(self.events),
                "attributes": len(self.attributes),
            },
        )
        self.snapshots.append(snapshot)
        return snapshot

    def latest_snapshot(self) -> Optional["KnowledgeSnapshot"]:
        return self.snapshots[-1] if self.snapshots else None

    def get_entities(self, refresh: bool = False) -> List[Dict[str, Any]]:
        if refresh or not self.entity_index:
            entities = self.neo4j_adapter.fetch_entities()
            self.entity_index = {entity["id"]: entity for entity in entities}
            self.alias_index.clear()
            for entity in entities:
                entity_id = entity.get("id")
                for alias in [entity.get("name", "")] + entity.get("aliases", []) or []:
                    if alias:
                        self.alias_index[alias.strip().lower()] = entity_id
        return list(self.entity_index.values())

    def get_relationships(self, refresh: bool = False) -> List[Dict[str, Any]]:
        if refresh or not self.relationship_index:
            relationships = self.neo4j_adapter.fetch_relationships()
            self.relationship_index = {rel["id"]: rel for rel in relationships}
        return list(self.relationship_index.values())

    def get_alias_index(self) -> Dict[str, str]:
        if not self.alias_index:
            self.get_entities(refresh=True)
        return dict(self.alias_index)

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def get_attributes(self) -> List[Dict[str, Any]]:
        return list(self.attributes)

    def get_communities(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.current_communities)

    def _build_community_profiles(
        self,
        communities: Dict[str, List[str]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not communities:
            return {}

        entity_index = {entity["id"]: entity for entity in entities}
        profiles: Dict[str, Dict[str, Any]] = {}

        graph = getattr(self.qa_engine, "graph", None)
        if graph is None:
            return {}

        relation_sets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for rel in relationships:
            source_id = rel.get("source_id") or rel.get("source")
            target_id = rel.get("target_id") or rel.get("target")
            if not source_id or not target_id:
                continue
            key = tuple(sorted((source_id, target_id)))
            relation_sets.setdefault(key, []).append(rel)

        for community_id, member_ids in communities.items():
            filtered_ids = [mid for mid in member_ids if mid in entity_index]
            if len(filtered_ids) < 2:
                continue

            member_entities = [entity_index[mid] for mid in filtered_ids]
            member_names = [entity.get("name") for entity in member_entities if entity.get("name")]
            member_set = set(filtered_ids)

            subgraph = graph.subgraph(filtered_ids).copy()
            edge_count = subgraph.number_of_edges()
            size = subgraph.number_of_nodes()
            density = 0.0
            if size > 1:
                try:
                    density = float(nx.density(subgraph))
                except Exception:
                    density = 0.0

            degree_scores = subgraph.degree()
            top_entities = sorted(
                (
                    {
                        "id": node,
                        "name": entity_index[node].get("name"),
                        "degree": degree,
                    }
                    for node, degree in degree_scores
                ),
                key=lambda item: item["degree"],
                reverse=True,
            )[:5]

            bridge_scores: Dict[str, int] = {}
            for node in filtered_ids:
                for neighbor in graph.neighbors(node):
                    if neighbor not in member_set:
                        bridge_scores[node] = bridge_scores.get(node, 0) + 1

            bridge_entities = [
                {
                    "id": node,
                    "name": entity_index[node].get("name"),
                    "external_degree": bridge_scores[node],
                }
                for node in sorted(bridge_scores, key=bridge_scores.get, reverse=True)[:5]
            ]

            key_relationships = []
            for source_id in filtered_ids:
                for target_id in filtered_ids:
                    if target_id <= source_id:
                        continue
                    rels = relation_sets.get(tuple(sorted((source_id, target_id))))
                    if not rels:
                        continue
                    for rel in rels:
                        key_relationships.append(
                            {
                                "source": entity_index[source_id].get("name"),
                                "target": entity_index[target_id].get("name"),
                                "relationship": rel.get("relationship"),
                                "description": rel.get("description"),
                            }
                        )
            key_relationships = key_relationships[:10]

            summary_payload = self.qa_engine.summarize_community(
                {
                    "community_id": community_id,
                    "member_preview": [
                        {
                            "id": entity.get("id"),
                            "name": entity.get("name"),
                            "type": entity.get("type"),
                            "description": entity.get("description"),
                        }
                        for entity in member_entities[:12]
                    ],
                    "top_entities": top_entities,
                    "bridge_entities": bridge_entities,
                    "key_relationships": key_relationships,
                    "metrics": {
                        "size": size,
                        "edge_count": edge_count,
                        "density": density,
                    },
                }
            )

            profile = {
                "members": filtered_ids,
                "member_names": member_names,
                "title": summary_payload.get("title") or f"社区 {community_id}",
                "summary": summary_payload.get("summary", ""),
                "keywords": summary_payload.get("keywords", []),
                "top_entities": top_entities,
                "bridge_entities": bridge_entities,
                "metrics": {
                    "size": size,
                    "edge_count": edge_count,
                    "density": density,
                },
            }

            profiles[community_id] = profile

        return profiles


class TemporalIndex:
    """记录知识快照的简单时间索引。"""

    def __init__(self) -> None:
        self.snapshots: List[KnowledgeSnapshot] = []

    def register_snapshot(self, snapshot: KnowledgeSnapshot) -> None:
        self.snapshots.append(snapshot)

    def timeline(self) -> List[Dict[str, Any]]:
        return [
            {
                "version": snapshot.version,
                "entity_count": snapshot.entity_count,
                "relationship_count": snapshot.relationship_count,
                "metadata": snapshot.metadata,
            }
            for snapshot in self.snapshots
        ]

    def latest(self) -> Optional[KnowledgeSnapshot]:
        return self.snapshots[-1] if self.snapshots else None


class EntityGraphView:
    """简化的实体图，用于保留接口兼容性。"""

    def __init__(self) -> None:
        self.graph = nx.Graph()

    def refresh(self, entities: Iterable[Dict[str, Any]], relationships: Iterable[Dict[str, Any]]) -> None:
        self.graph.clear()
        for entity in entities:
            node_id = entity.get("name")
            if node_id:
                self.graph.add_node(node_id, **entity)
        for relationship in relationships:
            source = relationship.get("source")
            target = relationship.get("target")
            if source and target:
                self.graph.add_edge(source, target, **relationship)


# -----------------
# 嵌入与存储适配器
# -----------------


class EmbeddingCache:
    """简单的向量缓存。"""

    def __init__(self) -> None:
        self._store: Dict[str, np.ndarray] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._store.get(key)

    def put(self, key: str, value: np.ndarray) -> None:
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()


class EmbeddingFabric:
    """统一的节点/关系嵌入生成器。"""

    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base") -> None:
        from advanced_vector_embedding import AdvancedVectorEmbedding

        self.embedder = AdvancedVectorEmbedding(model_name=model_name)
        self.cache = EmbeddingCache()

    def embed_entities(self, entities: Iterable[Dict[str, Any]], use_cache: bool = True) -> Dict[str, np.ndarray]:
        texts: List[str] = []
        ids: List[str] = []
        embeddings: Dict[str, np.ndarray] = {}

        for entity in entities:
            entity_id = entity.get("id") or entity.get("name")
            if not entity_id:
                continue
            cached = self.cache.get(entity_id) if use_cache else None
            if cached is not None:
                embeddings[entity_id] = cached
            else:
                ids.append(entity_id)
                texts.append(f"{entity.get('name')} | {entity.get('type')} | {entity.get('description')}")

        if texts:
            vectors = self.embedder.embed_texts(texts)
            for idx, entity_id in enumerate(ids):
                vector = vectors[idx]
                embeddings[entity_id] = vector
                self.cache.put(entity_id, vector)

        return embeddings

    def embed_relationships(self, relationships: Iterable[Dict[str, Any]], use_cache: bool = True) -> Dict[str, np.ndarray]:
        texts: List[str] = []
        ids: List[str] = []
        embeddings: Dict[str, np.ndarray] = {}

        for relationship in relationships:
            rel_id = relationship.get("id") or (
                f"{relationship.get('source')}::{relationship.get('relationship')}::{relationship.get('target')}"
            )
            if not rel_id:
                continue
            cached = self.cache.get(rel_id) if use_cache else None
            if cached is not None:
                embeddings[rel_id] = cached
            else:
                ids.append(rel_id)
                texts.append(f"{relationship.get('relationship')} | {relationship.get('description')}")

        if texts:
            vectors = self.embedder.embed_texts(texts)
            for idx, rel_id in enumerate(ids):
                vector = vectors[idx]
                embeddings[rel_id] = vector
                self.cache.put(rel_id, vector)

        return embeddings


class VectorAdapter:
    """内存向量存储。"""

    def __init__(self) -> None:
        self._store: Dict[str, np.ndarray] = {}

    def upsert(self, key: str, vector: np.ndarray) -> None:
        self._store[key] = vector

    def batch_upsert(self, payload: Dict[str, np.ndarray]) -> None:
        self._store.update(payload)

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> Dict[str, float]:
        scores = {}
        for key, vector in self._store.items():
            norm = np.linalg.norm(vector) * np.linalg.norm(query_vector)
            if norm == 0:
                continue
            scores[key] = float(vector.dot(query_vector) / norm)
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])

    def clear(self) -> None:
        self._store.clear()


class LocalStateStore:
    """以 JSON 文件持久化流水线状态。"""

    def __init__(self, path: str = "state_store.json") -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)


class Neo4jAdapter:
    """Neo4j 后端适配器，封装常用操作。"""

    def __init__(self, **kwargs: Any) -> None:
        from neo4j_storage import Neo4jKnowledgeGraph

        self.backend = Neo4jKnowledgeGraph(**kwargs)

    def upsert_entities(self, entities: List[Dict[str, Any]]) -> List[str]:
        if not entities:
            return []
        return self.backend.store_entities(entities, entity_embeddings=None)

    def upsert_relationships(self, relationships: List[Dict[str, Any]]) -> List[str]:
        if not relationships:
            return []
        return self.backend.store_relationships(relationships, relationship_embeddings=None)

    def fetch_entities(self) -> List[Dict[str, Any]]:
        with self.backend.driver.session(database=self.backend.database) as session:
            query = (
                "MATCH (e:Entity) "
                "RETURN e.id AS id, e.name AS name, e.type AS type, "
                "e.description AS description, e.domain AS domain, e.aliases AS aliases"
            )
            result = session.run(query)
            return [
                {
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "domain": record["domain"],
                    "aliases": record.get("aliases") or [],
                }
                for record in result
            ]

    def fetch_relationships(self) -> List[Dict[str, Any]]:
        with self.backend.driver.session(database=self.backend.database) as session:
            query = (
                "MATCH (source:Entity)-[r:RELATIONSHIP]->(target:Entity) "
                "RETURN r.id AS id, r.relationship AS relationship, r.description AS description, "
                "source.id AS source_id, source.name AS source_name, "
                "target.id AS target_id, target.name AS target_name"
            )
            result = session.run(query)
            return [
                {
                    "id": record["id"],
                    "relationship": record["relationship"],
                    "description": record["description"],
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "source": record["source_name"],
                    "target": record["target_name"],
                }
                for record in result
            ]

    def statistics(self) -> Dict[str, Any]:
        return self.backend.get_graph_statistics()

    def replace_communities(self, communities: Dict[str, Dict[str, Any]]) -> List[str]:
        return self.backend.replace_communities(communities)

    def store_embeddings(
        self,
        entity_embeddings: Optional[Dict[str, Any]] = None,
        relationship_embeddings: Optional[Dict[str, Any]] = None,
    ) -> None:
        entity_embeddings = entity_embeddings or {}
        relationship_embeddings = relationship_embeddings or {}

        if entity_embeddings:
            with self.backend.driver.session(database=self.backend.database) as session:
                session.run(
                    "UNWIND $payload AS item MATCH (e:Entity {id: item.id}) SET e.embedding = item.embedding",
                    payload=[
                        {
                            "id": key,
                            "embedding": vector.tolist() if hasattr(vector, "tolist") else vector,
                        }
                        for key, vector in entity_embeddings.items()
                    ],
                )

        if relationship_embeddings:
            with self.backend.driver.session(database=self.backend.database) as session:
                session.run(
                    "UNWIND $payload AS item MATCH ()-[r:RELATIONSHIP {id: item.id}]->() SET r.embedding = item.embedding",
                    payload=[
                        {
                            "id": key,
                            "embedding": vector.tolist() if hasattr(vector, "tolist") else vector,
                        }
                        for key, vector in relationship_embeddings.items()
                    ],
                )

    def run_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        parameters = parameters or {}
        with self.backend.driver.session(database=self.backend.database) as session:
            result = session.run(query, parameters)
            return [record.data() if hasattr(record, "data") else dict(record) for record in result]

    def close(self) -> None:
        self.backend.close()


# -----------------
# 评估与健康监控
# -----------------


class FeedbackLoop:
    """收集人工或自动化反馈的简单容器。"""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record(self, payload: Dict[str, Any]) -> None:
        self.records.append(payload)

    def summary(self) -> Dict[str, Any]:
        issues = [record for record in self.records if record.get("type") == "issue"]
        warnings = [record for record in self.records if record.get("type") == "warning"]
        infos = [record for record in self.records if record.get("type") == "info"]
        return {
            "count": len(self.records),
            "issues": issues,
            "warnings": warnings,
            "infos": infos,
        }


class KnowledgeMetrics:
    """基于 KnowledgeOS 输出指标。"""

    def __init__(self, knowledge_os: KnowledgeOS) -> None:
        self.knowledge_os = knowledge_os

    def snapshot_metrics(self) -> Dict[str, Any]:
        entities = self.knowledge_os.get_entities()
        relationships = self.knowledge_os.get_relationships()
        events = self.knowledge_os.get_events()
        attributes = self.knowledge_os.get_attributes()
        communities = self.knowledge_os.get_communities()
        return {
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "entity_types": len({entity.get("type") for entity in entities}),
            "relationship_types": len({rel.get("relationship") for rel in relationships}),
            "event_count": len(events),
            "attribute_count": len(attributes),
            "community_count": len(communities),
            "avg_relationship_degree": round(len(relationships) / max(len(entities), 1), 2),
        }


class KnowledgeHealthMonitor:
    """根据指标生成健康评分。"""

    def __init__(self, metrics: KnowledgeMetrics) -> None:
        self.metrics = metrics

    def health_report(self) -> Dict[str, Any]:
        snapshot = self.metrics.snapshot_metrics()

        density = snapshot["avg_relationship_degree"]
        coverage = 1.0 if snapshot["event_count"] > 0 or snapshot["attribute_count"] > 0 else 0.5

        community_profiles = self.metrics.knowledge_os.get_communities()
        if community_profiles:
            densities = [profile.get("metrics", {}).get("density", 0.0) for profile in community_profiles.values()]
            avg_community_density = sum(densities) / max(len(densities), 1)
            community_factor = min(1.0, 0.6 + avg_community_density * 0.4)
        else:
            avg_community_density = 0.0
            community_factor = 0.6

        score = min(1.0, 0.5 * min(1.0, density / 5.0) + 0.3 * coverage + 0.2 * community_factor)

        return {
            "snapshot": snapshot,
            "health_score": round(score, 2),
            "signals": {
                "density": density,
                "coverage": coverage,
                "communities": snapshot["community_count"],
                "community_density": round(avg_community_density, 3),
            },
        }


# -----------------
# 推理与回答组件
# -----------------


class IntentRouter:
    """使用 LLM 判别问题意图，失败时回退到关键词启发。"""

    def __init__(self, qa_engine: Optional[KnowledgeGraphQA] = None, enable_fallback: bool = True) -> None:
        self.qa_engine = qa_engine
        self.enable_fallback = enable_fallback

    def classify(self, question: str) -> Dict[str, Any]:
        if self.qa_engine:
            result = self.qa_engine.classify_intent(question)
            q_type = (result or {}).get("question_type")
            if q_type:
                return result

        if self.enable_fallback:
            return self._fallback_intent(question)
        return {"question_type": "descriptive", "raw": {}}

    def _fallback_intent(self, question: str) -> Dict[str, Any]:
        q_lower = question.lower()
        relation_keywords = ["合作伙伴", "partner", "合作方", "关系", "联盟"]
        attribute_keywords = ["多少", "占比", "指标", "性能", "优势", "能力", "价格"]
        causal_keywords = ["为什么", "原因", "导致", "影响", "why", "cause"]
        timeline_keywords = ["什么时候", "时间", "历程", "history", "发展史"]
        comparative_keywords = ["相比", "对比", "比较", "difference", "compare"]

        if any(keyword in q_lower for keyword in relation_keywords):
            q_type = "relation"
        elif any(keyword in q_lower for keyword in attribute_keywords):
            q_type = "attribute"
        elif any(keyword in q_lower for keyword in causal_keywords):
            q_type = "causal"
        elif any(keyword in q_lower for keyword in timeline_keywords):
            q_type = "temporal"
        elif any(keyword in q_lower for keyword in comparative_keywords):
            q_type = "comparative"
        else:
            q_type = "descriptive"
        return {"question_type": q_type, "raw": {"fallback": True}}


@dataclass
class QueryTemplate:
    name: str
    description: str
    cypher: str
    priority: int = 100
    weight: float = 1.0


class TemplateLibrary:
    """管理意图到 Cypher 模板的配置。"""

    def __init__(self, template_path: Optional[str] = None) -> None:
        self.template_path = template_path
        self.templates: Dict[str, List[QueryTemplate]] = {}
        if template_path:
            self.load(template_path)

    def load(self, template_path: str) -> None:
        path = Path(template_path)
        if not path.is_file():
            raise FileNotFoundError(f"未找到模板文件: {template_path}")
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        parsed: Dict[str, List[QueryTemplate]] = {}
        for intent, template_list in raw.items():
            parsed[intent] = []
            for entry in template_list or []:
                parsed[intent].append(
                    QueryTemplate(
                        name=entry.get("name", f"{intent}_template"),
                        description=entry.get("description", ""),
                        cypher=entry.get("cypher", ""),
                        priority=int(entry.get("priority", 100)),
                        weight=float(entry.get("weight", 1.0)),
                    )
                )
            parsed[intent].sort(key=lambda tpl: tpl.priority)

        self.template_path = template_path
        self.templates = parsed

    def get_templates(self, intent: str, fallback: str = "fallback") -> List[QueryTemplate]:
        return self.templates.get(intent, []) or self.templates.get(fallback, [])

    def intents(self) -> List[str]:
        return list(self.templates.keys())


class QueryPlanStep:
    def __init__(self, name: str, description: str, params: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.params = params


class QueryPlan:
    def __init__(self, intent: str, steps: List[QueryPlanStep]) -> None:
        self.intent = intent
        self.steps = steps


class QueryPlanner:
    """根据问题意图组装检索计划。"""

    def __init__(self, template_library: Optional[TemplateLibrary] = None) -> None:
        self.template_library = template_library

    def build_plan(
        self,
        intent: Dict[str, str],
        question: str,
        alias_index: Optional[Dict[str, str]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryPlan:
        q_type = intent.get("question_type", "descriptive")
        steps: List[QueryPlanStep] = []

        alias_index = alias_index or {}
        entities = entities or []
        focus_info = self._resolve_focus_entities(question, q_type, alias_index, entities)
        template_key = self._intent_to_template_key(q_type)

        templates: List[Dict[str, Any]] = []
        if self.template_library:
            for tpl in self.template_library.get_templates(template_key):
                templates.append(
                    {
                        "name": tpl.name,
                        "description": tpl.description,
                        "cypher": tpl.cypher,
                        "priority": tpl.priority,
                        "weight": tpl.weight,
                    }
                )

        steps.append(
            QueryPlanStep(
                "entity_lookup",
                "识别焦点实体",
                {
                    "focus_names": focus_info["focus_names"],
                    "focus_ids": focus_info["primary_ids"],
                },
            )
        )

        steps.append(QueryPlanStep("community_context", "获取社区摘要", {}))

        steps.append(
            QueryPlanStep(
                "cypher_retrieve",
                "执行图结构检索",
                {
                    "templates": templates,
                    "template_key": template_key,
                    "focus_ids": focus_info["primary_ids"] or focus_info["all_ids"],
                    "all_focus_ids": focus_info["all_ids"],
                    "secondary_ids": focus_info["secondary_ids"],
                    "focus_aliases": focus_info["focus_aliases"],
                    "relationship_keywords": focus_info["relationship_keywords"],
                    "context_terms": focus_info["context_terms"],
                    "limit": 10,
                },
            )
        )

        steps.append(
            QueryPlanStep(
                "vector_retrieve",
                "语义向量召回",
                {"top_k": 5 if q_type not in {"temporal", "attribute"} else 3},
            )
        )

        steps.append(QueryPlanStep("context_consistency", "一致性校验", {}))

        return QueryPlan(intent=q_type, steps=steps)

    def _intent_to_template_key(self, question_type: str) -> str:
        mapping = {
            "relation": "relation",
            "comparative": "comparative",
            "causal": "causal",
            "temporal": "timeline",
            "attribute": "attribute",
            "descriptive": "fallback",
        }
        return mapping.get(question_type, "fallback")

    def _resolve_focus_entities(
        self,
        question: str,
        question_type: str,
        alias_index: Dict[str, str],
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        q_lower = question.lower()
        matched_ids: List[str] = []

        for alias, entity_id in alias_index.items():
            if alias and alias in q_lower:
                matched_ids.append(entity_id)

        entity_lookup = {entity.get("id"): entity for entity in entities}
        if not matched_ids:
            for entity in entities:
                name = (entity.get("name") or "").lower()
                if name and name in q_lower:
                    matched_ids.append(entity.get("id"))

        matched_ids = [eid for eid in matched_ids if eid]
        seen: set[str] = set()
        focus_ids: List[str] = []
        for eid in matched_ids:
            if eid not in seen:
                focus_ids.append(eid)
                seen.add(eid)

        focus_ids = focus_ids[:3]
        primary_ids = focus_ids[:1]
        secondary_ids = focus_ids[1:]

        alias_terms: set[str] = set()
        focus_names: List[str] = []
        for eid in focus_ids:
            entity = entity_lookup.get(eid)
            if not entity:
                continue
            name = entity.get("name")
            if name:
                focus_names.append(name)
                alias_terms.add(name.lower())
            for alias in entity.get("aliases", []) or []:
                if alias:
                    alias_terms.add(alias.lower())

        if not alias_terms:
            for alias, eid in alias_index.items():
                if eid in focus_ids:
                    alias_terms.add(alias)

        context_terms = self._extract_context_terms(question)

        return {
            "primary_ids": primary_ids,
            "secondary_ids": secondary_ids,
            "all_ids": focus_ids,
            "focus_aliases": sorted(alias_terms),
            "focus_names": focus_names,
            "context_terms": context_terms,
            "relationship_keywords": [],
        }

    def _extract_context_terms(self, question: str) -> List[str]:
        tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_\-]+", question.lower()))
        return [token for token in tokens if len(token) > 1]


class PathExecutor:
    """根据检索计划执行图谱/向量混合检索。"""

    def __init__(
        self,
        knowledge_os: KnowledgeOS,
        entity_view,
        temporal_index,
        embedding_fabric: EmbeddingFabric,
        vector_store: VectorAdapter,
        neo4j_adapter: Optional[Neo4jAdapter] = None,
        relation_weights: Optional[Dict[str, float]] = None,
        qa_engine: Optional[KnowledgeGraphQA] = None,
        community_view=None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.knowledge_os = knowledge_os
        self.entity_view = entity_view
        self.temporal_index = temporal_index
        self.embedding_fabric = embedding_fabric
        self.vector_store = vector_store
        self.neo4j_adapter = neo4j_adapter
        self.relation_weights = relation_weights or {}
        self.qa_engine = qa_engine
        self.community_view = community_view

    def execute(self, plan: QueryPlan, question: str) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {"question": question, "plan": plan.intent, "steps": []}
        entities = self.knowledge_os.get_entities()
        matched_entities: List[Dict[str, Any]] = []

        for step in plan.steps:
            if step.name == "entity_lookup":
                matched = [entity for entity in entities if entity.get("name", "").lower() in question.lower()]
                evidence["steps"].append({"step": step.name, "entities": matched})
                matched_entities = matched

            elif step.name == "cypher_retrieve":
                context = self._cypher_retrieve(question, step.params, matched_entities)
                evidence["steps"].append({"step": step.name, "graph_paths": context})

            elif step.name == "vector_retrieve":
                retrieved = self._vector_neighbors(question, top_k=step.params.get("top_k", 5))
                evidence["steps"].append({"step": step.name, "vector_hits": retrieved})

            elif step.name == "community_context":
                context = self._community_context(matched_entities)
                if context:
                    evidence["steps"].append({"step": step.name, "communities": context})

            elif step.name == "timeline":
                timeline = self.temporal_index.timeline()
                evidence["steps"].append({"step": step.name, "timeline": timeline})

            elif step.name == "context_consistency":
                verdict = self._validate_consistency(evidence)
                evidence["steps"].append({"step": step.name, "verdict": verdict})

        return evidence

    def _cypher_retrieve(
        self,
        question: str,
        params: Dict[str, Any],
        matched_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.neo4j_adapter:
            return []

        limit = params.get("limit", 10)
        matched_entities = matched_entities or []
        templates = params.get("templates", []) or []
        focus_ids = params.get("focus_ids", []) or []
        all_focus_ids = params.get("all_focus_ids", []) or focus_ids
        secondary_ids = params.get("secondary_ids", []) or []
        focus_aliases = [alias.lower() for alias in params.get("focus_aliases", []) or []]
        relationship_keywords = [kw.lower() for kw in params.get("relationship_keywords", []) or []]
        context_terms = [term.lower() for term in params.get("context_terms", []) or []]

        results: List[Dict[str, Any]] = []
        if templates:
            payload_base = {
                "focus_ids": all_focus_ids or focus_ids or secondary_ids,
                "secondary_ids": secondary_ids,
                "all_aliases": focus_aliases,
                "relationship_keywords": relationship_keywords,
                "context_terms": context_terms,
                "limit": limit,
            }
            for template in templates:
                query = template.get("cypher")
                if not query:
                    continue
                payload = {k: v for k, v in payload_base.items() if v}
                try:
                    res = self.neo4j_adapter.run_cypher(query, payload)
                except Exception as exc:  # pragma: no cover - 防御
                    self.logger.warning("模板 %s 执行失败: %s", template.get("name"), exc)
                    continue
                if res:
                    normalized = self._normalize_records(res, template, payload_base)
                    results.extend(normalized)
                    break
            if results:
                return results

        search_terms: List[str] = []
        for entity in matched_entities:
            name = entity.get("name")
            if name:
                search_terms.append(name)
            for alias in entity.get("aliases", []) or []:
                if alias:
                    search_terms.append(alias)

        pattern = r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_\-]+"
        question_terms = [token for token in re.findall(pattern, question) if len(token.strip()) > 1]
        search_terms.extend(question_terms)
        search_terms.extend(context_terms)
        if relationship_keywords:
            search_terms.extend(relationship_keywords)

        unique_terms: List[str] = []
        seen_terms: set[str] = set()
        for term in search_terms:
            term_lower = term.lower()
            if term_lower and term_lower not in seen_terms:
                seen_terms.add(term_lower)
                unique_terms.append(term_lower)
        search_terms = unique_terms or [question]

        cypher = (
            "UNWIND $terms AS term MATCH path=(source:Entity)-[r:RELATIONSHIP]->(target:Entity) "
            "WHERE toLower(source.name) CONTAINS toLower(term) "
            "   OR toLower(target.name) CONTAINS toLower(term) "
            "   OR toLower(r.relationship) CONTAINS toLower(term) "
            "RETURN source.name AS source, target.name AS target, r.relationship AS relationship, r.description AS description LIMIT $limit"
        )

        records = self.neo4j_adapter.run_cypher(cypher, {"terms": search_terms[:10], "limit": limit})
        normalized = self._normalize_records(records, {"name": "fallback", "weight": 0.5}, {})
        if normalized:
            return normalized

        if self.qa_engine:
            generated = self._llm_generate_cypher(question, matched_entities, context_terms)
            if generated:
                combined: List[Dict[str, Any]] = []
                for query in generated:
                    if not self._is_safe_cypher(query):
                        continue
                    try:
                        res = self.neo4j_adapter.run_cypher(query, {"limit": limit})
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.warning("动态 Cypher 执行失败: %s", exc)
                        continue
                    combined.extend(self._normalize_records(res, {"name": "llm_generated", "weight": 0.6}, {}))
                if combined:
                    return combined

        return normalized

    def _vector_neighbors(self, question: str, top_k: int) -> Dict[str, float]:
        question_vector = self.embedding_fabric.embedder.embed_texts(question)[0]
        if question_vector is None:
            return {}
        return self.vector_store.query(question_vector, top_k=top_k)

    def _validate_consistency(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        graph_hits = []
        vector_hits = []
        for step in evidence.get("steps", []):
            if step.get("step") == "cypher_retrieve":
                graph_hits.extend(step.get("graph_paths", []))
            if step.get("step") == "vector_retrieve":
                vector_hits.append(step.get("vector_hits", {}))

        verdict = {
            "graph_hits": len(graph_hits),
            "vector_hits": len(vector_hits[0]) if vector_hits else 0,
            "status": "ok" if graph_hits or vector_hits else "insufficient",
        }
        return verdict

    def _normalize_records(
        self,
        records: List[Dict[str, Any]],
        template: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        weight = float(template.get("weight", 1.0))
        keywords = [kw for kw in payload.get("relationship_keywords", [])]
        focus_aliases = set(payload.get("all_aliases", []))

        for record in records or []:
            source_name = record.get("source_name") or record.get("source")
            target_name = record.get("target_name") or record.get("target")
            relationship = record.get("relationship") or record.get("relationship_type") or ""
            description = record.get("description") or record.get("event_description") or ""
            path_details = record.get("path_details")

            if not (source_name and target_name):
                continue

            base_relation = str(relationship or "").lower()
            score = weight + self.relation_weights.get(base_relation, 0.0)
            rel_lower = str(relationship).lower()
            for kw in keywords:
                if kw and kw in rel_lower:
                    score += 0.3
            if source_name.lower() in focus_aliases:
                score += 0.1
            if target_name.lower() in focus_aliases:
                score += 0.1

            if isinstance(path_details, list) and path_details:
                hops = []
                for detail in path_details:
                    mid = detail.get("mid")
                    rel1 = detail.get("rel1")
                    rel2 = detail.get("rel2")
                    hops.append(f"{mid}({rel1}->{rel2})")
                if hops:
                    description = description or " -> ".join(hops)
                    score += 0.1 * len(hops)

            normalized.append(
                {
                    "template": template.get("name"),
                    "source_name": source_name,
                    "target_name": target_name,
                    "relationship": relationship,
                    "description": description,
                    "path_details": path_details,
                    "score": round(score, 3),
                }
            )

        normalized.sort(key=lambda item: item.get("score", 0), reverse=True)
        return normalized

    def _community_context(self, matched_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not matched_entities:
            return []

        profiles = self.knowledge_os.get_communities() or {}
        if not profiles:
            return []

        entity_to_profile: Dict[str, Dict[str, Any]] = {}
        for community_id, profile in profiles.items():
            for member_id in profile.get("members", []):
                entity_to_profile[member_id] = {"community_id": community_id, **profile}

        context_entries: List[Dict[str, Any]] = []
        seen_ids = set()

        for entity in matched_entities:
            entity_id = entity.get("id")
            if not entity_id:
                continue
            profile = entity_to_profile.get(entity_id)
            if not profile:
                continue
            community_id = profile["community_id"]
            if community_id in seen_ids:
                continue
            seen_ids.add(community_id)
            context_entries.append(
                {
                    "community_id": community_id,
                    "title": profile.get("title"),
                    "summary": profile.get("summary"),
                    "keywords": profile.get("keywords", []),
                    "members": profile.get("member_names", [])[:10],
                    "top_entities": profile.get("top_entities", [])[:5],
                    "bridge_entities": profile.get("bridge_entities", [])[:5],
                }
            )

        return context_entries

    def _llm_generate_cypher(
        self,
        question: str,
        matched_entities: List[Dict[str, Any]],
        context_terms: List[str],
    ) -> List[str]:
        if not self.qa_engine:
            return []

        aliases = []
        for entity in matched_entities:
            aliases.append(entity.get("name", ""))
            aliases.extend(entity.get("aliases", []) or [])

        return self.qa_engine.generate_cypher_queries(
            question=question,
            focus_aliases=[alias for alias in aliases if alias],
            context_terms=context_terms,
        )

    def _is_safe_cypher(self, query: str) -> bool:
        unsafe = ["CREATE", "MERGE", "DELETE", "DETACH", "SET ", "DROP", "CALL dbms", "CALL apoc.create"]
        upper_query = query.upper()
        return not any(keyword in upper_query for keyword in unsafe)


class AnswerComposer:
    """综合证据并调用 LLM 生成最终回答。"""

    def __init__(self, qa_engine: KnowledgeGraphQA) -> None:
        self.qa_engine = qa_engine

    def compose(self, question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        context_lines, relevant_entities = self._build_evidence_context(evidence)
        context_text = "\n".join(context_lines) if context_lines else "暂无结构化证据。"

        question_analysis = self.qa_engine._analyze_question(question)
        answer_text = self.qa_engine._generate_answer(question, context_text, [])

        return {
            "answer": answer_text,
            "relevant_entities": relevant_entities,
            "question_analysis": question_analysis,
            "evidence_plan": evidence,
        }

    def _build_evidence_context(self, evidence: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        lines: List[str] = []
        entity_names: List[str] = []
        seen_entities: set[str] = set()

        for step in evidence.get("steps", []):
            if step.get("step") == "cypher_retrieve":
                for idx, path in enumerate(step.get("graph_paths", [])[:5], start=1):
                    source = path.get("source_name") or path.get("source")
                    target = path.get("target_name") or path.get("target")
                    relation = path.get("relationship", "关联") or "关联"
                    score = path.get("score")
                    description = path.get("description") or ""
                    snippet = f"{idx}. {source} --{relation}--> {target}"
                    if isinstance(score, (int, float)):
                        snippet += f" (score={score:.2f})"
                    if description:
                        snippet += f" | 描述: {description}"
                    lines.append(snippet)
                    for name in (source, target):
                        if name and name not in seen_entities:
                            seen_entities.add(name)
                            entity_names.append(name)

            if step.get("step") == "vector_retrieve":
                vector_hits = step.get("vector_hits", {})
                if vector_hits:
                    lines.append("向量检索提示：")
                    for name, score in list(vector_hits.items())[:5]:
                        lines.append(f"- {name} (sim={score:.2f})")

            if step.get("step") == "community_context":
                communities = step.get("communities", [])
                if communities:
                    lines.append("社区概览：")
                    for comm in communities:
                        title = comm.get("title") or comm.get("community_id")
                        summary = comm.get("summary") or ""
                        keywords = ", ".join(comm.get("keywords", [])[:5])
                        members = ", ".join(comm.get("members", [])[:6])
                        snippet = f"- {title}: {summary}"
                        if keywords:
                            snippet += f" | 关键词: {keywords}"
                        if members:
                            snippet += f" | 代表成员: {members}"
                        lines.append(snippet)
                        for member in comm.get("members", [])[:3]:
                            if member not in seen_entities:
                                seen_entities.add(member)
                                entity_names.append(member)

        return lines, entity_names


class AnswerCritic:
    """对回答做轻量审阅，标记潜在风险。"""

    def review(self, composed_answer: Dict[str, Any]) -> Dict[str, Any]:
        answer_text = composed_answer.get("answer", "")
        entities = composed_answer.get("relevant_entities", [])
        evidence = composed_answer.get("evidence_plan", {})
        concerns = []

        if not answer_text:
            concerns.append("empty_answer")
        if len(answer_text.split()) < 5:
            concerns.append("answer_too_short")
        if not entities:
            concerns.append("no_entities")

        steps = evidence.get("steps", []) if isinstance(evidence, dict) else []
        graph_hits: List[Dict[str, Any]] = []
        vector_hits: List[Dict[str, float]] = []
        for step in steps:
            if step.get("step") == "cypher_retrieve":
                graph_hits.extend(step.get("graph_paths", []))
            elif step.get("step") == "vector_retrieve":
                vector_hits.append(step.get("vector_hits", {}))

        has_graph = bool(graph_hits)
        has_vectors = bool(vector_hits)
        verdict_step = next((step for step in steps if step.get("step") == "context_consistency"), {})
        verdict_status = verdict_step.get("verdict", {}).get("status") if verdict_step else None

        if not has_graph and not has_vectors:
            concerns.append("no_evidence")
        if verdict_status == "insufficient":
            concerns.append("insufficient_context")

        approved = len(concerns) == 0
        confidence = "high" if approved else ("medium" if "no_evidence" not in concerns else "low")

        return {
            "concerns": concerns,
            "approved": approved,
            "confidence": confidence,
            "evidence_counts": {
                "graph_hits": len(graph_hits),
                "vector_hits": len(vector_hits[0]) if vector_hits else 0,
            },
        }


# -----------------
# 管道编排
# -----------------


class AuroraPipeline:
    """整合摄取、推理与评估的一站式管道。"""

    def __init__(
        self,
        config_path: str = "settings.yaml",
        embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
        use_neo4j: bool = True,
        state_store_path: str = "pipeline_state.json",
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path

        self.store = DocumentStore(config_path=config_path)
        self.router = DocumentRouter(self.store)

        chunk_cfg = self.store.config.get("chunking", {})
        self.chunker = HierarchicalChunker(
            max_chars=chunk_cfg.get("max_chars", 2000),
            min_chars=chunk_cfg.get("min_chars", 800),
            overlap_chars=chunk_cfg.get("overlap_chars", 200),
        )

        self.schema_manager = SchemaManager()

        template_cfg = self.store.config.get("templates", {})
        template_path = template_cfg.get("query")
        self.template_library: Optional[TemplateLibrary] = None
        if template_path:
            template_file = Path(template_path)
            if not template_file.is_absolute():
                template_file = Path(config_path).resolve().parent / template_file
            try:
                self.template_library = TemplateLibrary(str(template_file))
            except Exception as exc:
                self.logger.warning("加载查询模板失败 (%s)，将使用默认策略", exc)

        neo4j_config = self.store.config.get("neo4j", {})
        self.neo4j_adapter = Neo4jAdapter(
            uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            user=neo4j_config.get("user", "neo4j"),
            password=neo4j_config.get("password", "password"),
            database=neo4j_config.get("database", "neo4j"),
        )

        self.qa_engine = KnowledgeGraphQA(
            config_file=config_path,
            use_neo4j=use_neo4j,
            embedding_model=embedding_model,
            neo4j_uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            neo4j_user=neo4j_config.get("user", "neo4j"),
            neo4j_password=neo4j_config.get("password", "password"),
            neo4j_database=neo4j_config.get("database", "neo4j"),
        )

        self.knowledge_os = KnowledgeOS(
            qa_engine=self.qa_engine,
            schema_manager=self.schema_manager,
            neo4j_adapter=self.neo4j_adapter,
        )

        self.temporal_index = TemporalIndex()

        extraction_cfg = self.store.config.get("extraction", {})
        max_workers = extraction_cfg.get("max_workers", 4)

        alias_cfg = self.store.config.get("aliases", {})
        alias_dict_path = alias_cfg.get("dictionary")
        alias_dictionary: Dict[str, List[str]] = {}
        alias_file_path: Optional[Path] = None
        if alias_dict_path:
            alias_file = Path(alias_dict_path)
            if not alias_file.is_absolute():
                alias_file = Path(config_path).resolve().parent / alias_file
            alias_file_path = alias_file
            try:
                alias_dictionary = yaml.safe_load(alias_file.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                self.logger.warning("加载别名字典失败 (%s)，将不使用预置别名", exc)

        self.strategist = StrategistAgent(chunker=self.chunker, router=self.router)
        self.entity_agent = EntityRelationExtractionAgent(
            qa_engine=self.qa_engine,
            schema_manager=self.schema_manager,
            max_workers=max_workers,
            rate_limit=extraction_cfg.get("rate_limit", 0.3),
        )
        self.attribute_agent = AttributeExtractionAgent(self.qa_engine)
        self.event_agent = EventExtractionAgent(self.qa_engine)
        self.critic = ConsistencyCritic()
        self.curator = KnowledgeCuratorAgent()
        self.aggregator = AggregatorAgent(
            qa_engine=self.qa_engine,
            alias_dictionary=alias_dictionary,
            alias_dict_path=str(alias_file_path) if alias_file_path else None,
            max_workers=max_workers,
        )

        self.embedding_fabric = EmbeddingFabric(model_name=embedding_model)
        self.vector_store = VectorAdapter()
        self.entity_view = EntityGraphView()

        self.intent_router = IntentRouter(self.qa_engine)
        self.query_planner = QueryPlanner(template_library=self.template_library)
        self.path_executor = PathExecutor(
            knowledge_os=self.knowledge_os,
            entity_view=self.entity_view,
            temporal_index=self.temporal_index,
            embedding_fabric=self.embedding_fabric,
            vector_store=self.vector_store,
            neo4j_adapter=self.neo4j_adapter,
            relation_weights=self._relation_weights(),
            qa_engine=self.qa_engine,
        )
        self.answer_composer = AnswerComposer(self.qa_engine)
        self.answer_critic = AnswerCritic()

        self.feedback_loop = FeedbackLoop()
        self.metrics = KnowledgeMetrics(self.knowledge_os)
        self.health_monitor = KnowledgeHealthMonitor(self.metrics)

        self.state_store = LocalStateStore(state_store_path)

    def _relation_weights(self) -> Dict[str, float]:
        scoring_cfg = self.store.config.get("scoring", {})
        relation_weights = {
            key.lower(): float(value)
            for key, value in (scoring_cfg.get("relation_weights") or {}).items()
        }
        return relation_weights

    def _new_context(self) -> AgentContext:
        run_id = str(uuid.uuid4())
        return AgentContext(run_id=run_id)

    def ingest(self, corpus: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        context = self._new_context()
        previous_state = self.state_store.load()
        previous_communities = previous_state.get("community_profiles", {}) or {}

        tasks = self.strategist.run(context, corpus=corpus)
        extractions = self.entity_agent.run(context, tasks=tasks)
        attributes = self.attribute_agent.run(context, tasks=tasks)
        events = self.event_agent.run(context, tasks=tasks)

        aggregation = self.aggregator.run(
            context,
            extractions=extractions,
            attributes=attributes,
            events=events,
        )
        aggregated_extractions = aggregation.get("aggregated_extractions", [])
        aggregated_events = aggregation.get("events", [])
        aggregated_attributes = aggregation.get("attributes", [])

        critic_report = self.critic.run(context, extractions=aggregated_extractions)
        change_set = self.curator.run(context, extractions=aggregated_extractions, critic_report=critic_report)
        snapshot = self.knowledge_os.apply_change_set(change_set)
        self.temporal_index.register_snapshot(snapshot)

        entities = self.knowledge_os.get_entities()
        relationships = self.knowledge_os.get_relationships()
        self.entity_view.refresh(entities, relationships)

        entity_embeddings = self.embedding_fabric.embed_entities(entities)
        relationship_embeddings = self.embedding_fabric.embed_relationships(relationships)
        self.vector_store.batch_upsert(entity_embeddings)
        self.neo4j_adapter.store_embeddings(entity_embeddings=entity_embeddings, relationship_embeddings=relationship_embeddings)

        community_profiles = self.knowledge_os.get_communities()
        community_evolution = self._compute_community_evolution(previous_communities, community_profiles)

        if community_evolution.get("alerts"):
            self.feedback_loop.record(
                {
                    "type": "info",
                    "message": "community_evolution_alert",
                    "details": community_evolution["alerts"],
                }
            )

        self.state_store.save(
            {
                "latest_run": context.run_id,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "communities": len(community_profiles),
                "snapshot_version": snapshot.version,
                "community_profiles": community_profiles,
            }
        )

        metrics = self.metrics.snapshot_metrics()
        health_report = self.health_monitor.health_report()

        report = {
            "run_id": context.run_id,
            "snapshot_version": snapshot.version,
            "entities": metrics["entity_count"],
            "relationships": metrics["relationship_count"],
            "communities": len(community_profiles),
            "community_profiles": community_profiles,
            "community_evolution": community_evolution,
            "duplicates": critic_report.get("duplicates", {}),
            "events": len(aggregated_events),
            "attributes": len(aggregated_attributes),
            "health": health_report,
            "feedback": self.feedback_loop.summary(),
        }
        return report

    def _compute_community_evolution(
        self,
        previous: Dict[str, Dict[str, Any]],
        current: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous = previous or {}
        current = current or {}

        prev_ids = set(previous.keys())
        curr_ids = set(current.keys())

        new_ids = curr_ids - prev_ids
        removed_ids = prev_ids - curr_ids
        changed: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []

        for community_id in sorted(prev_ids & curr_ids):
            prev_profile = previous.get(community_id, {})
            curr_profile = current.get(community_id, {})
            prev_members = set(prev_profile.get("members", []))
            curr_members = set(curr_profile.get("members", []))
            added = sorted(curr_members - prev_members)
            removed = sorted(prev_members - curr_members)
            delta_size = len(curr_members) - len(prev_members)

            if added or removed or delta_size != 0:
                entry = {
                    "community_id": community_id,
                    "title": curr_profile.get("title") or prev_profile.get("title"),
                    "delta_size": delta_size,
                    "added_members": added[:10],
                    "removed_members": removed[:10],
                }
                changed.append(entry)
                if abs(delta_size) >= 3 or len(added) >= 3 or len(removed) >= 3:
                    alerts.append(entry)

        return {
            "new": sorted(new_ids),
            "removed": sorted(removed_ids),
            "changed": changed,
            "alerts": alerts,
        }

    def answer(self, question: str) -> Dict[str, Any]:
        intent = self.intent_router.classify(question)
        alias_index = self.knowledge_os.get_alias_index()
        entities = self.knowledge_os.get_entities()
        plan = self.query_planner.build_plan(intent, question, alias_index=alias_index, entities=entities)
        evidence = self.path_executor.execute(plan, question)
        composed = self.answer_composer.compose(question, evidence)
        review = self.answer_critic.review(composed)
        composed["critic"] = review
        return composed

    def health(self) -> Dict[str, Any]:
        return self.health_monitor.health_report()
