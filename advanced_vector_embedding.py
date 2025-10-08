#!/usr/bin/env python3
"""
向量嵌入模块
"""

import numpy as np
import logging
from typing import List, Union, Dict, Optional
import json
from pathlib import Path
import torch

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, falling back to basic embeddings")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedVectorEmbedding:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.recommended_models = {
            "chinese_best": "Alibaba-NLP/gte-multilingual-base",
            "multilingual_large": "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "chinese_efficient": "BAAI/bge-large-zh-v1.5",
            "multilingual_fast": "paraphrase-multilingual-MiniLM-L12-v2",
            "english_best": "Alibaba-NLP/gte-large-en-v1.5"
        }

        self._load_model()

    def _load_model(self):
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available")

            self.logger.info(f"正在加载模型: {self.model_name}")
            self.logger.info(f"使用设备: {self.device}")

            model_kwargs = {
                "trust_remote_code": True,
                "device": self.device
            }
            if "Qwen" in self.model_name or "gte-Qwen" in self.model_name:
                try:
                    self.model = SentenceTransformer(self.model_name, **model_kwargs)
                    if hasattr(self.model, 'max_seq_length'):
                        self.model.max_seq_length = min(8192, self.model.max_seq_length)
                except Exception as e:
                    self.logger.warning(f"Qwen模型加载失败，尝试备用模型: {e}")
                    self.model_name = self.recommended_models["chinese_efficient"]
                    self.model = SentenceTransformer(self.model_name, **model_kwargs)
            else:
                self.model = SentenceTransformer(self.model_name, **model_kwargs)

            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"模型加载成功，向量维度: {self.embedding_dim}")

            self._show_model_info()

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self._load_fallback_model()

    def _load_fallback_model(self):
        fallback_models = [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5"
        ]

        for model_name in fallback_models:
            try:
                self.logger.info(f"尝试加载备用模型: {model_name}")
                self.model = SentenceTransformer(model_name, device=self.device)
                self.model_name = model_name
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"备用模型加载成功: {model_name}, 维度: {self.embedding_dim}")
                return
            except Exception as e:
                self.logger.warning(f"备用模型 {model_name} 加载失败: {e}")
                continue

        raise Exception("所有模型都加载失败")

    def _show_model_info(self):
        try:
            self.logger.info("="*50)
            self.logger.info("模型信息:")
            self.logger.info(f"  名称: {self.model_name}")
            self.logger.info(f"  向量维度: {self.embedding_dim}")
            self.logger.info(f"  设备: {self.device}")

            if hasattr(self.model, 'max_seq_length'):
                self.logger.info(f"  最大序列长度: {self.model.max_seq_length}")

            try:
                model_size = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f"  参数数量: {model_size:,}")
            except:
                pass

            self.logger.info("="*50)
        except Exception as e:
            self.logger.warning(f"无法显示模型信息: {e}")

    def embed_texts(self, texts: Union[str, List[str]],
                   batch_size: int = 32,
                   normalize_embeddings: bool = True,
                   prompt_name: Optional[str] = None) -> np.ndarray:
        """
        将文本转换为向量嵌入

        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            normalize_embeddings: 是否归一化嵌入向量
            prompt_name: 提示名称（用于某些特定模型）

        Returns:
            numpy数组，形状为 (n_texts, embedding_dim)
        """
        if self.model is None:
            raise Exception("模型未加载")

        if isinstance(texts, str):
            texts = [texts]

        try:
            encode_kwargs = {
                "batch_size": batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": normalize_embeddings
            }

            if prompt_name and hasattr(self.model, 'encode'):
                encode_kwargs["prompt_name"] = prompt_name

            embeddings = self.model.encode(texts, **encode_kwargs)

            self.logger.info(f"文本嵌入完成，形状: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"文本嵌入失败: {e}")
            raise

    def compute_similarity(self, embeddings1: np.ndarray,
                         embeddings2: np.ndarray,
                         metric: str = "cosine") -> np.ndarray:
        """
        计算两组向量之间的相似度

        Args:
            embeddings1: 第一组向量
            embeddings2: 第二组向量
            metric: 相似度度量方法 ("cosine", "dot", "euclidean")

        Returns:
            相似度矩阵
        """
        try:
            if metric == "cosine":
                # 使用sentence-transformers内置的余弦相似度
                if hasattr(self.model, 'similarity'):
                    similarity_matrix = self.model.similarity(embeddings1, embeddings2)
                    return similarity_matrix.cpu().numpy() if hasattr(similarity_matrix, 'cpu') else similarity_matrix
                else:
                    # 手动计算余弦相似度
                    dot_product = np.dot(embeddings1, embeddings2.T)
                    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
                    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
                    similarity = dot_product / (norm1 * norm2.T + 1e-8)
                    return similarity

            elif metric == "dot":
                return np.dot(embeddings1, embeddings2.T)

            elif metric == "euclidean":
                from scipy.spatial.distance import cdist
                distances = cdist(embeddings1, embeddings2, metric='euclidean')
                return -distances

            else:
                raise ValueError(f"不支持的相似度度量方法: {metric}")

        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            return np.zeros((len(embeddings1), len(embeddings2)))

    def find_most_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5,
                         threshold: float = 0.0) -> List[tuple]:
        """
        找到与查询向量最相似的候选向量

        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选向量矩阵
            top_k: 返回前k个最相似的结果
            threshold: 相似度阈值

        Returns:
            包含(索引, 相似度分数)的列表
        """
        try:
            similarities = self.compute_similarity(
                query_embedding.reshape(1, -1),
                candidate_embeddings
            )[0]

            # 过滤低于阈值的相似度
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                return []

            # 获取top-k结果
            valid_similarities = similarities[valid_indices]
            top_indices = np.argsort(valid_similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                original_idx = valid_indices[idx]
                similarity_score = float(valid_similarities[idx])
                results.append((int(original_idx), similarity_score))

            return results

        except Exception as e:
            self.logger.error(f"最相似搜索失败: {e}")
            return []

    def embed_entities(self, entities: List[dict],
                      text_field: str = 'description',
                      include_name: bool = True) -> Dict[str, np.ndarray]:
        """
        为实体列表生成向量嵌入

        Args:
            entities: 实体列表，每个实体是字典
            text_field: 用于嵌入的文本字段
            include_name: 是否在文本中包含实体名称

        Returns:
            包含实体ID和对应嵌入的字典
        """
        entity_embeddings = {}

        texts = []
        entity_ids = []

        for entity in entities:
            entity_id = entity.get('id', entity.get('name', ''))

            text_parts = []
            if include_name and 'name' in entity:
                text_parts.append(entity['name'])
            if text_field in entity and entity[text_field]:
                text_parts.append(entity[text_field])

            if 'type' in entity:
                text_parts.append(f"类型: {entity['type']}")

            text = " ".join(text_parts).strip()

            if text:
                texts.append(text)
                entity_ids.append(entity_id)

        if texts:
            try:
                embeddings = self.embed_texts(texts)

                for entity_id, embedding in zip(entity_ids, embeddings):
                    entity_embeddings[entity_id] = embedding

                self.logger.info(f"成功为 {len(entity_embeddings)} 个实体生成向量嵌入")

            except Exception as e:
                self.logger.error(f"实体向量嵌入生成失败: {e}")

        return entity_embeddings

    def embed_relationships(self, relationships: List[dict]) -> Dict[str, np.ndarray]:
        """
        为关系列表生成向量嵌入

        Args:
            relationships: 关系列表

        Returns:
            包含关系ID和对应嵌入的字典
        """
        relationship_embeddings = {}

        texts = []
        relationship_ids = []

        for rel in relationships:
            rel_id = f"{rel.get('source', '')}-{rel.get('target', '')}-{rel.get('relationship', '')}"

            text_parts = []
            if rel.get('source'):
                text_parts.append(f"源实体: {rel['source']}")
            if rel.get('relationship'):
                text_parts.append(f"关系: {rel['relationship']}")
            if rel.get('target'):
                text_parts.append(f"目标实体: {rel['target']}")
            if rel.get('description'):
                text_parts.append(f"描述: {rel['description']}")

            text = " ".join(text_parts).strip()

            if text:
                texts.append(text)
                relationship_ids.append(rel_id)

        if texts:
            try:
                embeddings = self.embed_texts(texts)

                for rel_id, embedding in zip(relationship_ids, embeddings):
                    relationship_embeddings[rel_id] = embedding

                self.logger.info(f"成功为 {len(relationship_embeddings)} 个关系生成向量嵌入")

            except Exception as e:
                self.logger.error(f"关系向量嵌入生成失败: {e}")

        return relationship_embeddings

    def save_embeddings(self, embeddings: Dict[str, np.ndarray],
                       filename: str):
        """保存嵌入到文件"""
        try:
            serializable_embeddings = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in embeddings.items()
            }

            data = {
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "device": self.device,
                "embeddings": serializable_embeddings
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"嵌入已保存到: {filename}")

        except Exception as e:
            self.logger.error(f"嵌入保存失败: {e}")

    def load_embeddings(self, filename: str) -> Dict[str, np.ndarray]:
        """从文件加载嵌入"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            embeddings = {
                k: np.array(v) for k, v in data["embeddings"].items()
            }

            self.logger.info(f"嵌入已从 {filename} 加载")
            self.logger.info(f"模型信息: {data.get('model_name', 'unknown')}")

            return embeddings

        except Exception as e:
            self.logger.error(f"嵌入加载失败: {e}")
            return {}

    def benchmark_model(self, test_texts: List[str]) -> Dict[str, float]:
        """
        对模型进行基准测试

        Args:
            test_texts: 测试文本列表

        Returns:
            包含性能指标的字典
        """
        try:
            import time

            self.logger.info("开始模型基准测试...")

            start_time = time.time()
            embeddings = self.embed_texts(test_texts)
            encoding_time = time.time() - start_time

            start_time = time.time()
            similarities = self.compute_similarity(embeddings, embeddings)
            similarity_time = time.time() - start_time

            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

            metrics = {
                "encoding_time_per_text": encoding_time / len(test_texts),
                "similarity_computation_time": similarity_time,
                "average_similarity": float(avg_similarity),
                "embedding_dimension": self.embedding_dim,
                "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            }

            self.logger.info("基准测试完成:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")

            return metrics

        except Exception as e:
            self.logger.error(f"基准测试失败: {e}")
            return {}
