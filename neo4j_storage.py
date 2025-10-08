#!/usr/bin/env python3
"""
Neo4j知识图谱存储模块
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uuid
from neo4j import GraphDatabase, Driver, Session
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jKnowledgeGraph:
    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j"):
        """
        初始化Neo4j知识图谱存储

        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None

        self._connect()

    def _connect(self):
        """连接到Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            self.logger.info(f"成功连接到Neo4j数据库: {self.uri}")

            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                if result.single():
                    self.logger.info("数据库连接测试成功")

        except Exception as e:
            self.logger.error(f"Neo4j连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j连接已关闭")

    def clear_database(self):
        """清空数据库（谨慎使用）"""
        try:
            with self.driver.session(database=self.database) as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("数据库已清空")
        except Exception as e:
            self.logger.error(f"清空数据库失败: {e}")
            raise

    def create_constraints_and_indexes(self):
        """创建约束和索引以提高查询性能"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Relationship) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE"
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.domain)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Relationship) ON (r.relationship)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Relationship) ON (r.type)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Community) ON (c.name)",
            "CREATE FULLTEXT INDEX IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
            "CREATE FULLTEXT INDEX IF NOT EXISTS FOR (r:Relationship) ON EACH [r.description]"
        ]

        try:
            with self.driver.session(database=self.database) as session:
                # 创建约束
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        self.logger.info(f"约束创建成功: {constraint}")
                    except Exception as e:
                        self.logger.warning(f"约束创建失败（可能已存在）: {e}")

                # 创建索引
                for index in indexes:
                    try:
                        session.run(index)
                        self.logger.info(f"索引创建成功: {index}")
                    except Exception as e:
                        self.logger.warning(f"索引创建失败（可能已存在）: {e}")

        except Exception as e:
            self.logger.error(f"创建约束和索引失败: {e}")
            raise

    def store_entities(self, entities: List[Dict[str, Any]],
                      entity_embeddings: Optional[Dict[str, np.ndarray]] = None) -> List[str]:
        """
        存储实体到Neo4j

        Args:
            entities: 实体列表
            entity_embeddings: 实体向量嵌入

        Returns:
            存储的实体ID列表
        """
        stored_ids = []

        try:
            with self.driver.session(database=self.database) as session:
                for entity in entities:
                    # 确保实体有ID
                    entity_id = entity.get('id', str(uuid.uuid4()))

                    # 准备属性
                    aliases = entity.get('aliases') or []
                    if isinstance(aliases, set):
                        aliases = sorted(aliases)
                    properties = {
                        'id': entity_id,
                        'name': entity.get('name', ''),
                        'type': entity.get('type', 'UNKNOWN'),
                        'description': entity.get('description', ''),
                        'domain': entity.get('domain', 'unknown'),
                        'aliases': aliases,
                        'created_at': datetime.now().isoformat()
                    }

                    # 添加向量嵌入
                    if entity_embeddings and entity_id in entity_embeddings:
                        embedding = entity_embeddings[entity_id]
                        if isinstance(embedding, np.ndarray):
                            properties['embedding'] = embedding.tolist()
                        else:
                            properties['embedding'] = embedding
                    else:
                        properties['embedding'] = None

                    # 创建Cypher查询
                    query = """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.domain = $domain,
                        e.aliases = $aliases,
                        e.created_at = $created_at,
                        e.embedding = $embedding
                    RETURN e.id as id
                    """

                    result = session.run(query, **properties)
                    if result.single():
                        stored_ids.append(entity_id)

            self.logger.info(f"成功存储 {len(stored_ids)} 个实体")
            return stored_ids

        except Exception as e:
            self.logger.error(f"存储实体失败: {e}")
            raise

    def store_relationships(self, relationships: List[Dict[str, Any]],
                           relationship_embeddings: Optional[Dict[str, np.ndarray]] = None) -> List[str]:
        """
        存储关系到Neo4j

        Args:
            relationships: 关系列表
            relationship_embeddings: 关系向量嵌入

        Returns:
            存储的关系ID列表
        """
        stored_ids = []

        try:
            with self.driver.session(database=self.database) as session:
                for rel in relationships:
                    # 确保关系有ID
                    rel_id = rel.get('id', str(uuid.uuid4()))

                    # 获取源和目标实体ID
                    source_id = rel.get('source')
                    target_id = rel.get('target')

                    if not source_id or not target_id:
                        self.logger.warning(f"关系缺少源或目标ID: {rel}")
                        continue

                    # 准备关系属性
                    properties = {
                        'id': rel_id,
                        'source_id': source_id,
                        'target_id': target_id,
                        'relationship': rel.get('relationship', 'RELATED_TO'),
                        'description': rel.get('description', ''),
                        'type': rel.get('type', 'relationship'),
                        'created_at': datetime.now().isoformat()
                    }

                    # 添加向量嵌入
                    if relationship_embeddings and rel_id in relationship_embeddings:
                        embedding = relationship_embeddings[rel_id]
                        if isinstance(embedding, np.ndarray):
                            properties['embedding'] = embedding.tolist()
                        else:
                            properties['embedding'] = embedding
                    else:
                        properties['embedding'] = None

                    # 动态构建查询参数
                    query_parts = [
                        "MATCH (source:Entity {id: $source_id})",
                        "MATCH (target:Entity {id: $target_id})",
                        "MERGE (source)-[r:RELATIONSHIP {id: $id}]->(target)",
                        "SET r.relationship = $relationship,",
                        "r.description = $description,",
                        "r.type = $type,",
                        "r.created_at = $created_at"
                    ]

                    # 如果有嵌入，添加到SET语句
                    if 'embedding' in properties:
                        query_parts.append(", r.embedding = $embedding")

                    query_parts.append("RETURN r.id as id")
                    query = "\n".join(query_parts)

                    # 准备查询参数
                    query_params = {
                        'source_id': properties['source_id'],
                        'target_id': properties['target_id'],
                        'id': properties['id'],
                        'relationship': properties['relationship'],
                        'description': properties['description'],
                        'type': properties['type'],
                        'created_at': properties['created_at'],
                        'embedding': properties.get('embedding')
                    }

                    result = session.run(query, query_params)
                    if result.single():
                        stored_ids.append(rel_id)
                    else:
                        self.logger.warning(f"关系创建失败，可能源或目标实体不存在: {source_id} -> {target_id}")

            self.logger.info(f"成功存储 {len(stored_ids)} 个关系")
            return stored_ids

        except Exception as e:
            self.logger.error(f"存储关系失败: {e}")
            raise

    def store_communities(self, communities: Dict[str, Any]) -> List[str]:
        """
        存储社区信息到Neo4j

        Args:
            communities: 社区字典 {community_id: [entity_id_list]}

        Returns:
            存储的社区ID列表
        """
        stored_ids = []

        try:
            with self.driver.session(database=self.database) as session:
                for community_id, entity_ids in communities.items():
                    profile: Dict[str, Any]
                    if isinstance(entity_ids, dict):
                        profile = entity_ids
                        members = profile.get("members", [])
                        member_names = profile.get("member_names", [])
                        title = profile.get("title") or f"社区 {community_id}"
                        summary = profile.get("summary", "")
                        keywords = profile.get("keywords", [])
                        metrics = profile.get("metrics", {})
                        top_entities = profile.get("top_entities", [])
                        bridge_entities = profile.get("bridge_entities", [])
                    else:
                        profile = {}
                        members = entity_ids
                        member_names = []
                        title = f"社区 {community_id}"
                        summary = ""
                        keywords = []
                        metrics = {}
                        top_entities = []
                        bridge_entities = []

                    members = list(members)
                    member_names = list(member_names)

                    # 创建社区节点
                    query = """
                    MERGE (c:Community {id: $id})
                    SET c.name = $name,
                        c.description = $description,
                        c.summary = $summary,
                        c.keywords = $keywords,
                        c.member_ids = $member_ids,
                        c.member_names = $member_names,
                        c.metrics = $metrics,
                        c.top_entities = $top_entities,
                        c.bridge_entities = $bridge_entities,
                        c.updated_at = $updated_at,
                        c.created_at = coalesce(c.created_at, $created_at),
                        c.entity_count = $entity_count
                    RETURN c.id as id
                    """

                    now_iso = datetime.now().isoformat()

                    result = session.run(query, {
                        'id': community_id,
                        'name': title,
                        'description': f"包含 {len(members)} 个实体的社区",
                        'summary': summary,
                        'keywords': keywords,
                        'member_ids': members,
                        'member_names': member_names,
                        'metrics': metrics,
                        'top_entities': top_entities,
                        'bridge_entities': bridge_entities,
                        'updated_at': now_iso,
                        'created_at': now_iso,
                        'entity_count': len(members)
                    })

                    if result.single():
                        stored_ids.append(community_id)

                        # 为社区中的每个实体创建BELONGS_TO关系
                        for entity_id in members:
                            rel_query = """
                            MATCH (c:Community {id: $community_id})
                            MATCH (e:Entity {id: $entity_id})
                            MERGE (e)-[r:BELONGS_TO]->(c)
                            SET r.created_at = coalesce(r.created_at, $created_at),
                                r.updated_at = $updated_at
                            """

                            session.run(rel_query, {
                                'community_id': community_id,
                                'entity_id': entity_id,
                                'created_at': now_iso,
                                'updated_at': now_iso,
                            })

            self.logger.info(f"成功存储 {len(stored_ids)} 个社区")
            return stored_ids

        except Exception as e:
            self.logger.error(f"存储社区失败: {e}")
            raise

    def clear_communities(self) -> None:
        """删除所有社区节点及其关系。"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (e:Entity)-[r:BELONGS_TO]->(:Community) DELETE r")
                session.run("MATCH (c:Community) DETACH DELETE c")
            self.logger.info("已清空社区节点与BELONGS_TO关系")
        except Exception as e:
            self.logger.error(f"清理社区失败: {e}")
            raise

    def replace_communities(self, communities: Dict[str, List[str]]) -> List[str]:
        """用最新的社区划分替换数据库中的社区信息。"""
        self.clear_communities()
        if not communities:
            return []
        return self.store_communities(communities)

    def store_knowledge_graph(self, entities: List[Dict[str, Any]],
                            relationships: List[Dict[str, Any]],
                            communities: Dict[str, List[str]] = None,
                            entity_embeddings: Optional[Dict[str, np.ndarray]] = None,
                            relationship_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        存储完整的知识图谱

        Args:
            entities: 实体列表
            relationships: 关系列表
            communities: 社区信息
            entity_embeddings: 实体向量嵌入
            relationship_embeddings: 关系向量嵌入
        """
        try:
            self.logger.info("开始存储知识图谱到Neo4j...")

            # 创建约束和索引
            self.create_constraints_and_indexes()

            # 存储实体
            stored_entities = self.store_entities(entities, entity_embeddings)

            # 存储关系
            stored_relationships = self.store_relationships(relationships, relationship_embeddings)

            # 存储社区（如果提供）
            stored_communities = []
            if communities:
                stored_communities = self.store_communities(communities)

            self.logger.info("知识图谱存储完成")
            self.logger.info(f"实体: {len(stored_entities)}, 关系: {len(stored_relationships)}, 社区: {len(stored_communities)}")

            return {
                'entities': stored_entities,
                'relationships': stored_relationships,
                'communities': stored_communities
            }

        except Exception as e:
            self.logger.error(f"存储知识图谱失败: {e}")
            raise

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索实体

        Args:
            query: 搜索查询
            limit: 返回结果数量限制

        Returns:
            匹配的实体列表
        """
        try:
            with self.driver.session(database=self.database) as session:
                # 使用简单的名称匹配搜索
                cypher_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($search_query)
                   OR toLower(e.description) CONTAINS toLower($search_query)
                RETURN e.id as id, e.name as name, e.type as type,
                       e.description as description, e.domain as domain, 1.0 as score
                ORDER BY score DESC
                LIMIT $limit
                """

                result = session.run(cypher_query, search_query=query, limit=limit)

                entities = []
                for record in result:
                    entities.append({
                        'id': record['id'],
                        'name': record['name'],
                        'type': record['type'],
                        'description': record['description'],
                        'domain': record['domain'],
                        'score': record['score']
                    })

                return entities

        except Exception as e:
            self.logger.error(f"搜索实体失败: {e}")
            return []

    def find_entity_relationships(self, entity_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        查找实体的所有关系

        Args:
            entity_id: 实体ID
            limit: 返回结果数量限制

        Returns:
            关系列表
        """
        try:
            with self.driver.session(database=self.database) as session:
                cypher_query = """
                MATCH (source:Entity {id: $entity_id})-[r:RELATIONSHIP]->(target:Entity)
                RETURN r.id as id, target.id as target_id, target.name as target_name,
                       target.type as target_type, r.relationship as relationship,
                       r.description as description, 'outgoing' as direction
                UNION
                MATCH (source:Entity)-[r:RELATIONSHIP]->(target:Entity {id: $entity_id})
                RETURN r.id as id, source.id as source_id, source.name as source_name,
                       source.type as source_type, r.relationship as relationship,
                       r.description as description, 'incoming' as direction
                ORDER BY r.created_at DESC
                LIMIT $limit
                """

                result = session.run(cypher_query, entity_id=entity_id, limit=limit)

                relationships = []
                for record in result:
                    relationships.append({
                        'id': record['id'],
                        'relationship': record['relationship'],
                        'description': record['description'],
                        'direction': record['direction']
                    })

                    if record['direction'] == 'outgoing':
                        relationships[-1]['target_id'] = record['target_id']
                        relationships[-1]['target_name'] = record['target_name']
                        relationships[-1]['target_type'] = record['target_type']
                    else:
                        relationships[-1]['source_id'] = record['source_id']
                        relationships[-1]['source_name'] = record['source_name']
                        relationships[-1]['source_type'] = record['source_type']

                return relationships

        except Exception as e:
            self.logger.error(f"查找实体关系失败: {e}")
            return []

    def get_community_entities(self, community_id: str) -> List[Dict[str, Any]]:
        """
        获取社区中的所有实体

        Args:
            community_id: 社区ID

        Returns:
            实体列表
        """
        try:
            with self.driver.session(database=self.database) as session:
                cypher_query = """
                MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $community_id})
                RETURN e.id as id, e.name as name, e.type as type,
                       e.description as description, e.domain as domain
                ORDER BY e.name
                """

                result = session.run(cypher_query, community_id=community_id)

                entities = []
                for record in result:
                    entities.append({
                        'id': record['id'],
                        'name': record['name'],
                        'type': record['type'],
                        'description': record['description'],
                        'domain': record['domain']
                    })

                return entities

        except Exception as e:
            self.logger.error(f"获取社区实体失败: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息

        Returns:
            统计信息字典
        """
        try:
            with self.driver.session(database=self.database) as session:
                stats = {}

                # 节点数量
                result = session.run("MATCH (n) RETURN count(n) as count")
                stats['total_nodes'] = result.single()['count']

                # 实体数量
                result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                stats['entities'] = result.single()['count']

                # 关系数量
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats['total_relationships'] = result.single()['count']

                # 社区数量
                result = session.run("MATCH (c:Community) RETURN count(c) as count")
                stats['communities'] = result.single()['count']

                # 实体类型统计
                result = session.run("MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC")
                stats['entity_types'] = {record['type']: record['count'] for record in result}

                # 关系类型统计
                result = session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN r.relationship as type, count(r) as count ORDER BY count DESC")
                stats['relationship_types'] = {record['type']: record['count'] for record in result}

                return stats

        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}

    def export_graph(self, output_file: str):
        """
        导出图数据到JSON文件

        Args:
            output_file: 输出文件路径
        """
        try:
            with self.driver.session(database=self.database) as session:
                # 导出所有节点
                nodes_query = """
                MATCH (n)
                RETURN labels(n) as labels, properties(n) as properties
                """

                nodes_result = session.run(nodes_query)
                nodes = []
                for record in nodes_result:
                    nodes.append({
                        'labels': record['labels'],
                        'properties': dict(record['properties'])
                    })

                # 导出所有关系
                relationships_query = """
                MATCH (a)-[r]->(b)
                RETURN id(a) as source, labels(a) as source_labels,
                       id(b) as target, labels(b) as target_labels,
                       type(r) as type, properties(r) as properties
                """

                relationships_result = session.run(relationships_query)
                relationships = []
                for record in relationships_result:
                    relationships.append({
                        'source': str(record['source']),
                        'source_labels': record['source_labels'],
                        'target': str(record['target']),
                        'target_labels': record['target_labels'],
                        'type': record['type'],
                        'properties': dict(record['properties'])
                    })

                # 保存到文件
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'nodes': nodes,
                    'relationships': relationships
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                self.logger.info(f"图数据已导出到: {output_file}")
                self.logger.info(f"节点: {len(nodes)}, 关系: {len(relationships)}")

        except Exception as e:
            self.logger.error(f"导出图数据失败: {e}")
            raise
