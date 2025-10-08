#!/usr/bin/env python3
"""
知识图谱问答
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from openai import OpenAI
import re
from collections import defaultdict, Counter
import networkx as nx
from advanced_vector_embedding import AdvancedVectorEmbedding as VectorEmbedding
from neo4j_storage import Neo4jKnowledgeGraph

# 默认提示词，当提示词文件不可用时使用。
DEFAULT_ENTITY_EXTRACTION_PROMPT = """你是一位资深的知识工程师，擅长从复杂文本中发现和构建知识图谱。请深度分析以下文本，抽取出对构建知识图谱最有价值的实体和关系。

实体发现策略：
• 识别文本的核心主题和关键角色
• 发现具体的名词：人物、组织、系统、流程、技术等
• 抽取重要的概念和术语
• 实体类型应该反映其在知识域中的本质属性

关系洞察策略：
• 寻找实体间的依赖关系、层级关系、因果关系
• 识别协作关系和时间关系
• 关系命名应该准确反映语义本质

请严格按照以下JSON格式输出：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "实体类型",
      "description": "简洁但信息丰富的描述"
    }
  ],
  "relationships": [
    {
      "source": "源实体名称",
      "target": "目标实体名称",
      "relationship": "关系类型",
      "description": "关系的具体说明"
    }
  ]
}"""

DEFAULT_QUESTION_ANALYSIS_PROMPT = """分析用户问题，提取关键实体和问题类型。请按照以下JSON格式返回：
{
  "key_entities": ["实体1", "实体2"],
  "question_type": "定义/关系/比较/其他",
  "intent": "用户意图描述"
}"""

DEFAULT_ANSWER_GENERATION_PROMPT = """你是一个知识图谱问答助手。请根据提供的知识图谱信息回答用户问题。

回答要求：
1. 基于提供的上下文信息进行回答
2. 如果上下文中没有直接答案，请诚实地说明
3. 回答要准确、简洁、有条理
4. 可以适当推理，但不要超出已有信息的范围"""

DEFAULT_ENTITY_CONSOLIDATION_PROMPT = """你是一位企业知识图谱专家。判断两个实体名称是否指同一主体，仅输出JSON：{"same": true/false, "canonical": "可选规范名"}"""

DEFAULT_INTENT_CLASSIFIER_PROMPT = """你是一位资深问答编排专家。请分析问题的意图类别并给出JSON：
{
  "question_type": "relation/attribute/causal/temporal/comparative/descriptive",
  "key_entities": ["相关核心实体列表"],
  "rationale": "简要说明分类原因"
}
若无法判断，请将 question_type 设置为 "descriptive"。"""

DEFAULT_ATTRIBUTE_EXTRACTION_PROMPT = """你是一位企业知识抽取助手。请从文本中识别结构化指标或属性，并返回JSON：
{
  "attributes": [
    {
      "entity": "指向的核心实体",
      "attribute": "属性名称",
      "value": "属性值(字符串)",
      "unit": "可选单位",
      "attribute_type": "属性类型标签，可选",
      "evidence": "能够支撑该属性的原文片段"
    }
  ]
}
若没有属性，请返回 {"attributes": []}。"""

DEFAULT_EVENT_EXTRACTION_PROMPT = """你是一位事件抽取专家。请识别文本中的关键事件并返回JSON：
{
  "events": [
    {
      "type": "事件类型，如 EVENT/ANNOUNCEMENT等",
      "label": "事件简短标题",
      "participants": ["涉及实体列表"],
      "timestamp": "可选时间表达(若无返回null)",
      "context": "支撑该事件的原文片段"
    }
  ]
}
若没有事件，请返回 {"events": []}。"""

DEFAULT_RELATIONSHIP_NORMALIZATION_PROMPT = """你是一位知识图谱关系标准化专家。请根据给定信息输出统一的关系类型标签，返回JSON：
{
  "canonical": "规范化的关系标签，建议使用简洁名词或动词短语",
  "confidence": 0-1之间的小数，可选
}
如果你无法确定，请保持原始关系词。禁止输出额外文本。"""

DEFAULT_RELATIONSHIP_EQUIVALENCE_PROMPT = """你是一位知识图谱关系对齐专家。判断两个关系标签是否表达同一种语义。
输入：
{
  "source": "源实体名称",
  "target": "目标实体名称",
  "relation_a": "关系A",
  "relation_b": "关系B",
  "context_a": "关系A描述，可为空",
  "context_b": "关系B描述，可为空"
}
输出 JSON：{"same": true/false}
若两者语义一致或近似，请返回 true；否则 false。不要附加其他文本。"""

DEFAULT_RELATION_WEIGHT_PROMPT = """你是一位图谱建模专家，需要为不同关系标签分配重要性权重。
请基于每个关系的业务意义、可能的上下文价值，输出 JSON：
{
  "关系名": 权重 (0~1 之间的小数)
}
整体权重不必归一化，但需体现相对优先级；若无法判断，请给出 0.2。
请覆盖所有传入的关系名称，禁止多余文本。"""

DEFAULT_COMMUNITY_SUMMARY_PROMPT = """你是一位企业知识图谱专家，请根据输入的社区成员信息生成简洁摘要。
请返回 JSON：
{
  "title": "一句话概括（不超过12个汉字）",
  "summary": "50~80字的社区说明，突出主题与关键成员",
  "keywords": ["3~6个关键词"]
}
若信息不足，可用"暂无数据"作为 title/summary。禁止输出额外文本。"""


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeGraphQA:
    def __init__(self, config_file: str = "settings.yaml",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 neo4j_database: str = "neo4j",
                 embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
                 use_neo4j: bool = False):
        """
        初始化知识图谱问答系统

        Args:
            config_file: 配置文件路径
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: Neo4j数据库名
            embedding_model: 向量嵌入模型名称
            use_neo4j: 是否使用Neo4j存储
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_file).resolve().parent
        self.prompts = {}
        self.load_config(config_file)

        # 初始化DeepSeek客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

        # 初始化向量嵌入模型
        self.vector_embedder = VectorEmbedding(embedding_model)

        # Neo4j存储
        self.use_neo4j = use_neo4j
        self.neo4j_storage = None
        if use_neo4j:
            try:
                self.neo4j_storage = Neo4jKnowledgeGraph(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                    database=neo4j_database
                )
                self.logger.info("Neo4j存储初始化成功")
            except Exception as e:
                self.logger.warning(f"Neo4j初始化失败，将使用内存存储: {e}")
                self.use_neo4j = False

        # 存储知识图谱数据
        self.entities = {}
        self.relationships = []
        self.communities = {}
        self.entity_embeddings = {}
        self.relationship_embeddings = {}

        # 构建图结构
        self.graph = nx.Graph()

    def load_config(self, config_file: str):
        """加载配置文件"""
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            self.config = config

            active_model = config.get('active_model', 'deepseek_chat')
            model_config = config['models'][active_model]

            self.api_key = model_config['api_key']
            self.api_base = model_config['api_base']
            self.model_name = model_config['model_name']
            self.max_tokens = model_config['max_tokens']

            self.logger.info(f"配置加载成功: {active_model}")

            self._load_prompts()

        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise

    def _load_prompts(self) -> None:
        prompt_config = (getattr(self, "config", {}) or {}).get("prompts", {})

        self.prompts = {
            "entity_extraction": self._load_prompt_file(
                prompt_config.get("entity_extraction_system", "prompts/entity_extraction_system.txt"),
                DEFAULT_ENTITY_EXTRACTION_PROMPT,
            ),
            "question_analysis": self._load_prompt_file(
                prompt_config.get("question_analysis_system", "prompts/question_analysis_system.txt"),
                DEFAULT_QUESTION_ANALYSIS_PROMPT,
            ),
            "answer_generation": self._load_prompt_file(
                prompt_config.get("answer_generation_system", "prompts/answer_generation_system.txt"),
                DEFAULT_ANSWER_GENERATION_PROMPT,
            ),
            "cypher_generation": self._load_prompt_file(
                prompt_config.get("cypher_generation_system", "prompts/cypher_generation_system.txt"),
                ""
            ),
            "entity_consolidation": self._load_prompt_file(
                prompt_config.get("entity_consolidation_system", "prompts/entity_consolidation_system.txt"),
                DEFAULT_ENTITY_CONSOLIDATION_PROMPT,
            ),
            "intent_classifier": self._load_prompt_file(
                prompt_config.get("intent_classifier_system", "prompts/intent_classifier_system.txt"),
                DEFAULT_INTENT_CLASSIFIER_PROMPT,
            ),
            "attribute_extraction": self._load_prompt_file(
                prompt_config.get("attribute_extraction_system", "prompts/attribute_extraction_system.txt"),
                DEFAULT_ATTRIBUTE_EXTRACTION_PROMPT,
            ),
            "event_extraction": self._load_prompt_file(
                prompt_config.get("event_extraction_system", "prompts/event_extraction_system.txt"),
                DEFAULT_EVENT_EXTRACTION_PROMPT,
            ),
            "relationship_normalization": self._load_prompt_file(
                prompt_config.get("relationship_normalization_system", "prompts/relationship_normalization_system.txt"),
                DEFAULT_RELATIONSHIP_NORMALIZATION_PROMPT,
            ),
            "relationship_equivalence": self._load_prompt_file(
                prompt_config.get("relationship_equivalence_system", "prompts/relationship_equivalence_system.txt"),
                DEFAULT_RELATIONSHIP_EQUIVALENCE_PROMPT,
            ),
            "community_summary": self._load_prompt_file(
                prompt_config.get("community_summary_system", "prompts/community_summary_system.txt"),
                DEFAULT_COMMUNITY_SUMMARY_PROMPT,
            ),
            "relation_weighting": self._load_prompt_file(
                prompt_config.get("relation_weighting_system", "prompts/relation_weighting_system.txt"),
                DEFAULT_RELATION_WEIGHT_PROMPT,
            ),
        }

    def _load_prompt_file(self, relative_path: str, default: str) -> str:
        if not relative_path:
            return default

        prompt_path = Path(relative_path)
        if not prompt_path.is_absolute():
            base_dir = getattr(self, "config_dir", Path.cwd())
            prompt_path = (base_dir / prompt_path).resolve()

        try:
            content = prompt_path.read_text(encoding='utf-8').strip()
            if content:
                return content
            self.logger.warning("提示词文件 %s 为空，使用默认内容", prompt_path)
        except Exception as exc:
            self.logger.warning("提示词文件 %s 读取失败，使用默认内容: %s", prompt_path, exc)
        return default

    def extract_entities_and_relationships(self, text: str) -> Tuple[List[dict], List[dict]]:
        """
        从文本中抽取实体和关系

        Args:
            text: 输入文本

        Returns:
            实体列表和关系列表的元组
        """
        try:
            messages = [
                {"role": "system", "content": self.prompts.get("entity_extraction", DEFAULT_ENTITY_EXTRACTION_PROMPT)},
                {"role": "user", "content": f"请分析以下文本并抽取实体和关系：\n\n{text}"}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            entities = result.get('entities', [])
            relationships = result.get('relationships', [])

            # 清理和验证数据
            entities = self._clean_entities(entities)
            relationships = self._clean_relationships(relationships, entities)

            self.logger.info(f"抽取结果 - 实体: {len(entities)}, 关系: {len(relationships)}")
            return entities, relationships

        except Exception as e:
            self.logger.error(f"实体和关系抽取失败: {e}")
            return [], []

    def _clean_entities(self, entities: List[dict]) -> List[dict]:
        """清理实体数据"""
        cleaned_entities = []
        seen_names = set()

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            name = str(entity.get('name', '')).strip()
            entity_type = str(entity.get('type', '')).strip()
            description = str(entity.get('description', '')).strip()

            if not name or name in seen_names:
                continue

            seen_names.add(name)
            cleaned_entities.append({
                'name': name,
                'type': entity_type or 'CONCEPT',
                'description': description or f"{name}实体"
            })

        return cleaned_entities

    def _clean_relationships(self, relationships: List[dict], entities: List[dict]) -> List[dict]:
        """清理关系数据"""
        cleaned_relationships = []
        entity_names = {entity['name'] for entity in entities}

        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            source = str(rel.get('source', '')).strip()
            target = str(rel.get('target', '')).strip()
            relationship = str(rel.get('relationship', '')).strip()
            description = str(rel.get('description', '')).strip()

            # 验证实体是否存在
            if source not in entity_names or target not in entity_names:
                continue

            if source == target:
                continue

            cleaned_relationships.append({
                'source': source,
                'target': target,
                'relationship': relationship or 'RELATED_TO',
                'description': description or f"{source}与{target}之间的{relationship}关系"
            })

        return cleaned_relationships

    def build_knowledge_graph(
        self,
        entities: List[dict],
        relationships: List[dict],
        *,
        sync_to_neo4j: bool | None = None,
    ):
        """
        构建知识图谱

        Args:
            entities: 实体列表
            relationships: 关系列表
        """
        # 清空现有数据
        self.entities = {}
        self.relationships = []
        self.graph.clear()

        # 添加实体
        used_ids = set()
        name_to_id: Dict[str, str] = {}

        for i, entity in enumerate(entities):
            preferred_id = entity.get('id') or f"entity_{i}"
            entity_id = preferred_id
            suffix = 1
            while entity_id in used_ids:
                entity_id = f"{preferred_id}_{suffix}"
                suffix += 1
            used_ids.add(entity_id)

            aliases = entity.get('aliases', [])
            self.entities[entity_id] = {
                'id': entity_id,
                'name': entity.get('name', f"实体_{i}"),
                'type': entity.get('type', 'CONCEPT'),
                'description': entity.get('description', f"{entity.get('name', '实体')}实体"),
                'aliases': aliases,
            }

            self.graph.add_node(entity_id, **self.entities[entity_id])
            name_to_id[self.entities[entity_id]['name']] = entity_id
            for alias in aliases:
                if alias and alias not in name_to_id:
                    name_to_id[alias] = entity_id

        # 添加关系
        for rel in relationships:
            source_id = rel.get('source_id') or name_to_id.get(rel.get('source'))
            target_id = rel.get('target_id') or name_to_id.get(rel.get('target'))

            if source_id and target_id:
                relationship_data = {
                    'id': f"rel_{len(self.relationships)}",
                    'source': source_id,
                    'target': target_id,
                    'relationship': rel.get('relationship', 'RELATED_TO'),
                    'description': rel.get('description', '')
                }
                self.relationships.append(relationship_data)

                self.graph.add_edge(
                    source_id,
                    target_id,
                    relationship=rel.get('relationship'),
                    description=rel.get('description')
                )

        # 生成向量嵌入
        self._generate_embeddings()

        # 如果启用Neo4j，存储到数据库
        if sync_to_neo4j is None:
            sync_to_neo4j = self.use_neo4j and self.neo4j_storage is not None

        if sync_to_neo4j and self.neo4j_storage:
            try:
                self.neo4j_storage.store_knowledge_graph(
                    entities=list(self.entities.values()),
                    relationships=self.relationships,
                    entity_embeddings=self.entity_embeddings,
                    relationship_embeddings=self.relationship_embeddings
                )
                self.logger.info("知识图谱已保存到Neo4j数据库")
            except Exception as e:
                self.logger.error(f"Neo4j存储失败: {e}")

        self.logger.info(
            "知识图谱构建完成 - 节点: %s, 边: %s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _generate_embeddings(self):
        """生成实体和关系的向量嵌入"""
        try:
            # 实体嵌入
            entity_texts = [f"{entity['name']} {entity['type']} {entity['description']}"
                           for entity in self.entities.values()]
            entity_embeddings = self.vector_embedder.embed_texts(entity_texts)

            for i, (entity_id, _) in enumerate(self.entities.items()):
                self.entity_embeddings[entity_id] = entity_embeddings[i]

            # 关系嵌入
            relationship_texts = [f"{rel['relationship']} {rel['description']}"
                                for rel in self.relationships]
            if relationship_texts:
                relationship_embeddings = self.vector_embedder.embed_texts(relationship_texts)
                for i, rel in enumerate(self.relationships):
                    rel['embedding'] = relationship_embeddings[i].tolist()

            self.logger.info("向量嵌入生成完成")

        except Exception as e:
            self.logger.error(f"向量嵌入生成失败: {e}")

    def detect_communities(self) -> Dict[str, List[str]]:
        """
        检测知识图谱中的社区结构

        Returns:
            社区字典
        """
        try:
            if self.graph.number_of_nodes() == 0:
                return {}

            # 使用NetworkX的Louvain算法
            communities = nx.community.louvain_communities(self.graph, resolution=1.0)

            self.communities = {
                f"community_{i}": list(community)
                for i, community in enumerate(communities)
                if len(community) >= 2  # 过滤掉太小的社区
            }

            self.logger.info(f"检测到 {len(self.communities)} 个社区")
            return self.communities

        except Exception as e:
            self.logger.error(f"社区检测失败: {e}")
            return {}

    def answer_question(self, question: str) -> dict:
        """
        基于知识图谱回答问题

        Args:
            question: 用户问题

        Returns:
            回答结果字典
        """
        try:
            # 1. 理解问题并识别关键实体
            question_analysis = self._analyze_question(question)

            # 2. 在知识图谱中搜索相关信息
            relevant_entities = self._search_relevant_entities(question_analysis, question)

            # 3. 构建上下文
            context = self._build_context(relevant_entities, question)

            # 4. 生成回答
            answer = self._generate_answer(question, context, relevant_entities)

            return {
                'question': question,
                'answer': answer,
                'relevant_entities': [self.entities[eid]['name'] for eid in relevant_entities],
                'question_analysis': question_analysis
            }

        except Exception as e:
            self.logger.error(f"问题回答失败: {e}")
            return {
                'question': question,
                'answer': f"抱歉，我无法回答这个问题: {str(e)}",
                'relevant_entities': [],
                'question_analysis': {}
            }

    def _analyze_question(self, question: str) -> dict:
        """分析问题，提取关键信息"""
        try:
            messages = [
                {"role": "system", "content": self.prompts.get("question_analysis", DEFAULT_QUESTION_ANALYSIS_PROMPT)},
                {"role": "user", "content": f"分析问题: {question}"}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"问题分析失败: {e}")
            return {"key_entities": [], "question_type": "其他", "intent": "无法分析"}

    def _search_relevant_entities(self, question_analysis: dict, question: str) -> List[str]:
        """搜索与问题相关的实体"""
        relevant_entities = []
        key_entities = question_analysis.get('key_entities', [])

        # 精确匹配
        for entity_name in key_entities:
            for entity_id, entity in self.entities.items():
                if entity_name.lower() in entity['name'].lower():
                    relevant_entities.append(entity_id)

        # 如果没有精确匹配，使用语义搜索
        if not relevant_entities and self.entity_embeddings:
            try:
                # 将问题转换为向量
                question_embedding = self.vector_embedder.embed_texts(question)

                # 计算与所有实体的相似度
                entity_ids = list(self.entity_embeddings.keys())
                entity_vectors = np.array([self.entity_embeddings[eid] for eid in entity_ids])

                # 确保数据类型一致
                question_embedding = question_embedding.astype(np.float64)
                entity_vectors = entity_vectors.astype(np.float64)

                similarities = self.vector_embedder.compute_similarity(
                    question_embedding, entity_vectors
                )[0]

                # 获取最相似的实体
                top_indices = np.argsort(similarities)[::-1][:3]  # 取前3个
                relevant_entities = [entity_ids[i] for i in top_indices if similarities[i] > 0.3]

            except Exception as e:
                self.logger.error(f"语义搜索失败: {e}")

        return relevant_entities

    def _build_context(self, relevant_entities: List[str], question: str) -> str:
        """构建回答的上下文"""
        context_parts = []

        # 如果使用Neo4j且有相关实体，从Neo4j获取更丰富的信息
        if self.use_neo4j and self.neo4j_storage and relevant_entities:
            try:
                for entity_id in relevant_entities:
                    if entity_id in self.entities:
                        entity_name = self.entities[entity_id]['name']
                        # 从Neo4j搜索相关实体和关系
                        related_entities = self.neo4j_storage.search_entities(entity_name, limit=5)
                        for related_entity in related_entities:
                            context_parts.append(f"实体: {related_entity['name']} (类型: {related_entity['type']}) - {related_entity.get('description', '无描述')}")
            except Exception as e:
                self.logger.warning(f"从Neo4j获取上下文失败: {e}")

        # 添加本地实体信息
        for entity_id in relevant_entities:
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                context_parts.append(f"实体: {entity['name']} ({entity['type']}) - {entity['description']}")

        # 添加本地关系信息
        for rel in self.relationships:
            if rel['source'] in relevant_entities or rel['target'] in relevant_entities:
                source_name = self.entities[rel['source']]['name']
                target_name = self.entities[rel['target']]['name']
                context_parts.append(f"关系: {source_name} {rel['relationship']} {target_name} - {rel['description']}")

        # 如果没有本地数据但有Neo4j，直接使用Neo4j的信息
        if not context_parts and self.use_neo4j and self.neo4j_storage:
            try:
                # 从问题中提取关键词进行搜索
                search_results = self.neo4j_storage.search_entities(question.replace("目前本地知识图谱中有哪些", "").replace("实体和关系", "").strip(), limit=10)
                if search_results:
                    context_parts.append("从Neo4j数据库中找到以下实体:")
                    for entity in search_results:
                        context_parts.append(f"实体: {entity['name']} (类型: {entity['type']}) - {entity.get('description', '无描述')}")
            except Exception as e:
                self.logger.warning(f"Neo4j搜索失败: {e}")

        return "\n".join(context_parts) if context_parts else "没有找到相关的直接信息。"

    def _generate_answer(self, question: str, context: str, relevant_entities: List[str]) -> str:
        """基于上下文生成回答"""
        try:
            messages = [
                {"role": "system", "content": self.prompts.get("answer_generation", DEFAULT_ANSWER_GENERATION_PROMPT)},
                {"role": "user", "content": f"问题: {question}\n\n知识图谱信息:\n{context}\n\n请基于以上信息回答问题。"}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"回答生成失败: {e}")
            return "抱歉，我无法生成回答。"

    def classify_intent(self, question: str) -> Dict[str, Any]:
        prompt = self.prompts.get("intent_classifier", DEFAULT_INTENT_CLASSIFIER_PROMPT)
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question.strip()},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=600,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            question_type = str(payload.get("question_type", "descriptive")).lower()
            allowed = {"relation", "attribute", "causal", "temporal", "comparative", "descriptive"}
            if question_type not in allowed:
                question_type = "descriptive"
            return {
                "question_type": question_type,
                "raw": payload,
            }
        except Exception as exc:
            self.logger.warning("Intent classification fallback due to error: %s", exc)
            return {
                "question_type": "descriptive",
                "raw": {"error": str(exc)},
            }

    def extract_attributes(self, text: str) -> List[Dict[str, Any]]:
        prompt = self.prompts.get("attribute_extraction", DEFAULT_ATTRIBUTE_EXTRACTION_PROMPT)
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text.strip()},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            attributes: List[Dict[str, Any]] = []
            for item in payload.get("attributes", []):
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity") or item.get("entity_name") or "").strip()
                attribute_name = str(item.get("attribute") or item.get("name") or "").strip()
                value = str(item.get("value") or "").strip()
                if not (entity and attribute_name and value):
                    continue
                attributes.append(
                    {
                        "entity": entity,
                        "attribute": attribute_name,
                        "value": value,
                        "unit": str(item.get("unit") or "").strip(),
                        "attribute_type": str(item.get("attribute_type") or "ATTRIBUTE").strip() or "ATTRIBUTE",
                        "evidence": str(item.get("evidence") or "").strip(),
                    }
                )
            return attributes
        except Exception as exc:
            self.logger.error("属性抽取失败: %s", exc)
            return []

    def extract_events(self, text: str) -> List[Dict[str, Any]]:
        prompt = self.prompts.get("event_extraction", DEFAULT_EVENT_EXTRACTION_PROMPT)
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text.strip()},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            events: List[Dict[str, Any]] = []
            for item in payload.get("events", []):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or item.get("title") or "").strip()
                context = str(item.get("context") or "").strip()
                if not (label and context):
                    continue
                events.append(
                    {
                        "type": str(item.get("type") or "EVENT").strip() or "EVENT",
                        "label": label,
                        "participants": [p.strip() for p in item.get("participants", []) if isinstance(p, str) and p.strip()],
                        "timestamp": str(item.get("timestamp") or "").strip() or None,
                        "context": context,
                    }
                )
            return events
        except Exception as exc:
            self.logger.error("事件抽取失败: %s", exc)
            return []

    def normalize_relationship_label(
        self,
        source: str,
        target: str,
        relationship: str,
        description: str | None = None,
    ) -> Dict[str, Any]:
        prompt = self.prompts.get("relationship_normalization", DEFAULT_RELATIONSHIP_NORMALIZATION_PROMPT)
        payload = {
            "source": source or "",
            "target": target or "",
            "relationship": relationship or "",
            "description": description or "",
        }

        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = json.loads(response.choices[0].message.content)
            canonical = content.get("canonical") or relationship or "RELATED_TO"
            return {
                "canonical": canonical,
                "confidence": content.get("confidence"),
            }
        except Exception as exc:
            self.logger.warning("关系标准化失败: %s", exc)
            return {"canonical": relationship or "RELATED_TO"}

    def compare_relationship_labels(
        self,
        source: str,
        target: str,
        relation_a: str,
        relation_b: str,
        context_a: str | None = None,
        context_b: str | None = None,
    ) -> Dict[str, Any]:
        prompt = self.prompts.get("relationship_equivalence", DEFAULT_RELATIONSHIP_EQUIVALENCE_PROMPT)
        payload = {
            "source": source or "",
            "target": target or "",
            "relation_a": relation_a or "",
            "relation_b": relation_b or "",
            "context_a": context_a or "",
            "context_b": context_b or "",
        }
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            self.logger.warning("关系等价判断失败: %s", exc)
            return {"same": False}

    def summarize_community(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompts.get("community_summary", DEFAULT_COMMUNITY_SUMMARY_PROMPT)
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            if isinstance(data, dict):
                data.setdefault("keywords", [])
                return data
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Community summary failed: %s", exc)
        return {"title": "社区概览", "summary": "暂无社区摘要", "keywords": []}

    def generate_relation_weights(self, relations: List[str]) -> Dict[str, float]:
        prompt = self.prompts.get("relation_weighting", DEFAULT_RELATION_WEIGHT_PROMPT)
        unique = sorted({rel for rel in relations if rel})
        if not unique:
            return {}

        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"relations": unique}, ensure_ascii=False)},
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=600,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            weights: Dict[str, float] = {}
            for rel in unique:
                value = payload.get(rel)
                try:
                    weights[rel] = float(value)
                except Exception:
                    weights[rel] = 0.2
            return weights
        except Exception as exc:  # pragma: no cover - 防御
            self.logger.warning("Relation weighting prompt failed: %s", exc)
            return {}

    def generate_cypher_queries(
        self,
        question: str,
        focus_aliases: List[str] | None = None,
        context_terms: List[str] | None = None,
    ) -> List[str]:
        if not self.prompts.get("cypher_generation"):
            return []

        payload = {
            "question": question,
            "focus_aliases": focus_aliases or [],
            "context_terms": context_terms or [],
        }

        prompt_user = (
            "问题: {question}\n"
            "已知实体别名: {aliases}\n"
            "上下文关键词: {terms}\n"
            "请生成至多三条可执行的只读Cypher查询。"
        ).format(
            question=question,
            aliases=", ".join(payload["focus_aliases"]) or "无",
            terms=", ".join(payload["context_terms"]) or "无",
        )

        try:
            messages = [
                {"role": "system", "content": self.prompts.get("cypher_generation")},
                {"role": "user", "content": prompt_user},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=600,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = json.loads(response.choices[0].message.content)
            queries = content.get("queries", [])
            return [
                query.strip() for query in queries
                if isinstance(query, str) and query.strip()
            ][:3]
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Cypher generation failed: %s", exc)
            return []

    def confirm_entity_equivalence(
        self,
        entity_a: str,
        entity_b: str,
        context: str | None = None,
    ) -> Dict[str, Any]:
        prompt = self.prompts.get("entity_consolidation")
        if not prompt:
            return {"same": False}

        user_content = (
            "实体A: {a}\n实体B: {b}\n上下文: {ctx}"
        ).format(
            a=entity_a,
            b=entity_b,
            ctx=context or "无",
        )

        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            payload = json.loads(response.choices[0].message.content)
            return payload if isinstance(payload, dict) else {"same": False}
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Entity consolidation prompt failed: %s", exc)
            return {"same": False}

    def process_document(self, text: str) -> dict:
        """
        处理文档并构建完整的知识图谱

        Args:
            text: 输入文档文本

        Returns:
            处理结果统计
        """
        try:
            self.logger.info("开始处理文档...")

            # 1. 抽取实体和关系
            entities, relationships = self.extract_entities_and_relationships(text)

            if not entities:
                return {"error": "未能从文本中抽取到任何实体"}

            # 2. 构建知识图谱
            self.build_knowledge_graph(entities, relationships)

            # 3. 检测社区
            communities = self.detect_communities()

            # 4. 返回统计信息
            stats = {
                "entities_count": len(self.entities),
                "relationships_count": len(self.relationships),
                "communities_count": len(communities),
                "entities": list(self.entities.values()),
                "relationships": self.relationships,
                "communities": communities
            }

            self.logger.info(f"文档处理完成 - 实体: {stats['entities_count']}, 关系: {stats['relationships_count']}")
            return stats

        except Exception as e:
            self.logger.error(f"文档处理失败: {e}")
            return {"error": str(e)}

    def save_knowledge_graph(self, filename: str = "knowledge_graph.json"):
        """保存知识图谱到文件"""
        try:
            kg_data = {
                "entities": list(self.entities.values()),
                "relationships": self.relationships,
                "communities": self.communities,
                "entity_embeddings": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in self.entity_embeddings.items()}
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(kg_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"知识图谱已保存到: {filename}")

        except Exception as e:
            self.logger.error(f"知识图谱保存失败: {e}")

    def load_knowledge_graph(self, filename: str = None):
        """从文件或Neo4j加载知识图谱"""
        try:
            # 优先从Neo4j加载
            if self.use_neo4j and self.neo4j_storage:
                self.logger.info("正在从Neo4j加载知识图谱...")
                try:
                    # 获取Neo4j中的统计信息
                    stats = self.neo4j_storage.get_graph_statistics()
                    if stats.get('entities', 0) > 0:
                        self.logger.info(f"Neo4j中发现 {stats.get('entities', 0)} 个实体，从Neo4j加载")

                        # 这里简化处理，直接使用Neo4j的数据
                        # 在实际使用中，问答会直接查询Neo4j
                        self.logger.info("知识图谱已从Neo4j连接建立，将直接使用Neo4j进行查询")
                        return
                except Exception as e:
                    self.logger.warning(f"从Neo4j加载失败: {e}，尝试从文件加载")

            # 如果Neo4j不可用或没有数据，从文件加载
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)

                # 重建实体和关系
                self.entities = {entity['id']: entity for entity in kg_data['entities']}
                self.relationships = kg_data['relationships']
                self.communities = kg_data.get('communities', {})
                self.entity_embeddings = kg_data.get('entity_embeddings', {})

                # 重建图结构
                self.graph.clear()
                for entity_id, entity in self.entities.items():
                    self.graph.add_node(entity_id, **entity)

                for rel in self.relationships:
                    self.graph.add_edge(
                        rel['source'],
                        rel['target'],
                        relationship=rel['relationship'],
                        description=rel['description']
                    )

                self.logger.info(f"知识图谱已从 {filename} 加载")
            else:
                self.logger.warning("没有找到可加载的知识图谱数据源")

        except Exception as e:
            self.logger.error(f"知识图谱加载失败: {e}")
