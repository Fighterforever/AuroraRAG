# AuroraRAG

AuroraRAG 是一个基于知识图谱的智能问答系统。

## 系统架构

### 核心特性

- **知识图谱构建**：自动从文档中抽取实体、关系、属性和事件
- **语义理解**：基于大语言模型的实体合并、关系标准化
- **社区检测**：自动识别知识社区，生成社区画像
- **智能问答**：多策略混合检索，支持图路径、向量和Cypher查询
- **分布式存储**：Neo4j 图数据库 + 本地向量缓存

### 系统组件

```
AuroraRag/
├── aurora_core.py              # 核心引擎（代理、图谱、推理、编排）
├── knowledge_graph_qa.py       # LLM 抽取与问答引擎
├── advanced_vector_embedding.py # 向量模型封装
├── neo4j_storage.py            # Neo4j 适配器
├── web_server.py               # FastAPI 服务接口
├── main.py                     # CLI 入口
├── interfaces/
│   └── cli.py                  # 命令行接口实现
├── prompts/                    # LLM 提示词模板
├── templates/                  # Cypher 查询模板
├── frontend/                   # React 前端界面
├── sample_data/                # 示例数据
├── scripts/                    # 运维脚本
└── settings.yaml               # 系统配置文件
```

## 快速开始

### 环境要求

- Python 3.10+
- Neo4j 5.0+ 
- Node.js 18+ (如需使用前端)

### 安装步骤

1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

2. 启动 Neo4j 数据库

使用 Docker Compose (推荐):

```bash
docker compose -f docker/docker-compose.yml up -d
```

或使用已有的 Neo4j 实例，需配置连接信息到 `settings.yaml`。

3. 配置系统

编辑 `settings.yaml`，配置以下关键参数：

```yaml
models:
  deepseek_chat:
    api_key: "你的API密钥"
    api_base: "https://api.deepseek.com"
    model_name: "deepseek-chat"

neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
  database: "neo4j"
```

### 基础使用

1. 摄取文档构建知识图谱

```bash
python main.py ingest
```

系统会自动：
- 读取 `sample_data/` 目录下的文档
- 抽取实体、关系、属性和事件
- 检测知识社区并生成画像
- 将数据存储到 Neo4j

2. 命令行问答

```bash
# 单次问答
python main.py qa "拉赫玛尼诺夫的主要作品有哪些？"

# 多轮对话模式
python main.py qa
# 输入问题，输入 exit 或 quit 退出
```

3. 完整流程

```bash
# 摄取后立即进入问答模式
python main.py full
```

## API 服务

### 启动服务

```bash
uvicorn web_server:app --reload --port 8000
```

服务启动后可访问 `http://localhost:8000/docs` 查看 API 文档。

### API 接口

#### 1. 摄取接口

**POST** `/ingest`

执行文档摄取，构建知识图谱。

**请求示例**：

```bash
curl -X POST http://localhost:8000/ingest
```

**响应示例**：

```json
{
  "report": {
    "entity_count": 28,
    "relationship_count": 35,
    "event_count": 12,
    "attribute_count": 15,
    "communities": 3,
    "health_score": 0.82,
    "feedback": []
  }
}
```

**响应字段说明**：

- `entity_count`: 实体总数
- `relationship_count`: 关系总数
- `event_count`: 事件总数
- `attribute_count`: 属性总数
- `communities`: 社区数量
- `health_score`: 知识图谱健康度评分 (0-1)
- `feedback`: 系统反馈信息（如社区演化告警）

#### 2. 问答接口

**POST** `/query`

基于知识图谱回答问题。

**请求参数**：

```json
{
  "question": "用户问题"
}
```

**请求示例**：

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"拉赫玛尼诺夫的主要作品有哪些？"}'
```

**响应示例**：

```json
{
  "answer": {
    "answer": "拉赫玛尼诺夫的主要作品包括：1) 四部钢琴协奏曲，其中第二钢琴协奏曲最为著名；2) 《帕格尼尼主题狂想曲》，其中的第18变奏广为传唱；3) 三部交响曲；4) 多部交响诗如《死之岛》、《交响舞曲》；5) 钢琴独奏作品如G小调前奏曲和第2号钢琴奏鸣曲；6) 改编作品包括巴赫前奏曲、里姆斯基-科萨科夫的《大黄蜂的飞行》等。",
    "relevant_entities": ["拉赫玛尼诺夫", "第二钢琴协奏曲", "帕格尼尼主题狂想曲", "死之岛", "交响舞曲"],
    "question_analysis": {
      "question_type": "attribute",
      "key_entities": ["拉赫玛尼诺夫"],
      "intent": "查询作曲家作品"
    },
    "evidence_plan": {
      "steps": [
        {
          "step": "cypher_retrieve",
          "graph_paths": [
            {
              "source_name": "拉赫玛尼诺夫",
              "target_name": "第二钢琴协奏曲",
              "relationship": "创作",
              "description": "1900年完成第二钢琴协奏曲"
            },
            {
              "source_name": "拉赫玛尼诺夫",
              "target_name": "帕格尼尼主题狂想曲",
              "relationship": "创作",
              "description": "创作帕格尼尼主题狂想曲"
            }
          ]
        },
        {
          "step": "community_context",
          "communities": [
            {
              "community_id": "community_0",
              "title": "俄国作曲家音乐社区",
              "summary": "该社区以拉赫玛尼诺夫为核心，包含俄国音乐传统和浪漫主义晚期音乐风格...",
              "keywords": ["俄国音乐", "浪漫主义", "钢琴协奏曲", "交响曲"]
            }
          ]
        }
      ]
    },
    "critic": {
      "approved": true,
      "confidence": "high",
      "concerns": []
    }
  }
}
```

**响应字段说明**：

- `answer`: 生成的答案文本
- `relevant_entities`: 相关实体列表
- `question_analysis`: 问题分析结果
  - `question_type`: 问题类型 (relation/attribute/causal/temporal/comparative/descriptive)
  - `key_entities`: 关键实体
  - `intent`: 问题意图
- `evidence_plan`: 证据检索计划
  - `steps`: 执行步骤列表
    - `cypher_retrieve`: Cypher 查询结果
    - `vector_retrieve`: 向量检索结果
    - `community_context`: 社区上下文
    - `context_consistency`: 一致性检查
- `critic`: 答案质量评估
  - `approved`: 是否通过
  - `confidence`: 置信度 (high/medium/low)
  - `concerns`: 潜在问题列表

#### 3. 健康检查接口

**GET** `/health`

获取知识图谱健康度报告。

**请求示例**：

```bash
curl http://localhost:8000/health
```

**响应示例**：

```json
{
  "snapshot": {
    "entity_count": 28,
    "relationship_count": 35,
    "event_count": 12,
    "attribute_count": 15,
    "community_count": 3,
    "avg_relationship_degree": 2.5
  },
  "health_score": 0.82,
  "signals": {
    "density": 2.5,
    "coverage": 1.0,
    "communities": 3,
    "community_density": 0.42
  }
}
```

**响应字段说明**：

- `snapshot`: 当前快照
  - `entity_count`: 实体数量
  - `relationship_count`: 关系数量
  - `event_count`: 事件数量
  - `attribute_count`: 属性数量
  - `community_count`: 社区数量
  - `avg_relationship_degree`: 平均关系度
- `health_score`: 综合健康评分 (0-1)
- `signals`: 健康信号
  - `density`: 图密度
  - `coverage`: 覆盖度
  - `communities`: 社区数
  - `community_density`: 社区平均密度

## Web 前端

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认运行在 `http://localhost:5173`。

### 功能特性

- 三栏式布局
  - 左侧：会话管理
  - 中间：问答交互区
  - 右侧：社区洞察面板
- 实时展示社区摘要、关键词、核心成员
- 支持多轮对话
- 玻璃拟态 UI 设计

## 配置说明

### settings.yaml 配置项

#### 模型配置

```yaml
models:
  deepseek_chat:
    api_base: "https://api.deepseek.com"
    api_key: "your-api-key"
    model_name: "deepseek-chat"
    temperature: 0.1
    max_tokens: 8192
active_model: deepseek_chat
```

#### 文档路径配置

```yaml
paths:
  target_directory: ./sample_data
  progress_file: extraction_progress.json
  raw_results_dir: raw_extraction_results
  output_prefix: GraphRAG分析结果
```

#### 分块配置

```yaml
chunking:
  max_chars: 2000      # 最大块大小
  min_chars: 800       # 最小块大小
  overlap_chars: 200   # 重叠字符数
```

#### 抽取配置

```yaml
extraction:
  max_workers: 4       # 并发工作线程数
  rate_limit: 0.3      # API调用频率限制（秒）
```

#### 关系权重配置

```yaml
scoring:
  relation_weights:
    领投: 0.4
    合作: 0.3
    合作伙伴: 0.35
    战略合作: 0.5
```

#### Neo4j 配置

```yaml
neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: password
  database: neo4j
```

## 运维脚本

### 重置数据库

```bash
# 清空 Neo4j 数据库
python scripts/reset_neo4j.py --use-docker

# 同时清理本地缓存文件
python scripts/reset_neo4j.py --use-docker --clean-local
```

### 导出数据

```bash
# 导出 Neo4j 数据到 JSON
python scripts/export_neo4j.py
```

导出的数据将保存在 `graph_export.json`。

## 核心类说明

### AuroraPipeline

管道编排类，提供统一的数据处理接口。

**主要方法**：

```python
from aurora_core import AuroraPipeline

pipeline = AuroraPipeline()

# 摄取文档
report = pipeline.ingest()

# 问答
answer = pipeline.answer("你的问题")

# 健康检查
health = pipeline.health()
```

### KnowledgeGraphQA

知识图谱问答引擎，负责 LLM 交互和知识抽取。

**主要方法**：

```python
from knowledge_graph_qa import KnowledgeGraphQA

qa = KnowledgeGraphQA(
    config_file="settings.yaml",
    use_neo4j=True,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# 抽取实体和关系
entities, relationships = qa.extract_entities_and_relationships(text)

# 构建知识图谱
qa.build_knowledge_graph(entities, relationships, sync_to_neo4j=True)

# 回答问题
result = qa.answer_question("问题")
```

### Neo4jKnowledgeGraph

Neo4j 存储适配器。

**主要方法**：

```python
from neo4j_storage import Neo4jKnowledgeGraph

storage = Neo4jKnowledgeGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# 存储实体
storage.store_entities(entities, entity_embeddings)

# 存储关系
storage.store_relationships(relationships)

# 存储社区
storage.store_communities(communities)

# 搜索实体
results = storage.search_entities("搜索关键词", limit=10)

# 获取统计信息
stats = storage.get_graph_statistics()
```

### AdvancedVectorEmbedding

向量嵌入模型封装。

**主要方法**：

```python
from advanced_vector_embedding import AdvancedVectorEmbedding

embedder = AdvancedVectorEmbedding(
    model_name="Alibaba-NLP/gte-multilingual-base"
)

# 文本嵌入
embeddings = embedder.embed_texts(["文本1", "文本2"])

# 计算相似度
similarity = embedder.compute_similarity(embeddings1, embeddings2)

# 查找最相似
results = embedder.find_most_similar(
    query_embedding,
    candidate_embeddings,
    top_k=5,
    threshold=0.3
)
```

## 提示词管理

所有 LLM 提示词位于 `prompts/` 目录：

- `entity_extraction_system.txt` - 实体抽取提示词
- `relationship_normalization_system.txt` - 关系标准化提示词
- `relationship_equivalence_system.txt` - 关系等价判断提示词
- `entity_consolidation_system.txt` - 实体合并提示词
- `attribute_extraction_system.txt` - 属性抽取提示词
- `event_extraction_system.txt` - 事件抽取提示词
- `question_analysis_system.txt` - 问题分析提示词
- `answer_generation_system.txt` - 答案生成提示词
- `cypher_generation_system.txt` - Cypher 查询生成提示词
- `community_summary_system.txt` - 社区摘要生成提示词
- `intent_classifier_system.txt` - 意图分类提示词

可根据业务需求修改这些提示词以优化效果。

## 常见问题

### 1. Neo4j 连接失败

检查 Neo4j 服务是否正常运行：

```bash
docker ps | grep neo4j
```

确认 `settings.yaml` 中的连接参数正确。

### 2. API 密钥配置

编辑 `settings.yaml`，配置 DeepSeek API 密钥：

```yaml
models:
  deepseek_chat:
    api_key: "sk-your-api-key"
```

### 3. 向量模型下载慢

首次运行时，系统会自动下载向量模型。可预先下载模型到 HuggingFace 缓存目录。

### 4. 内存不足

调整配置文件中的并发参数：

```yaml
extraction:
  max_workers: 2 
```

## 性能优化

### 并发优化

目前实现的多级并发：

1. 文档分段并发抽取
2. 关系标准化并发处理
3. 实体合并并发比较
4. 使用 `as_completed` 提高响应性

### 缓存机制

1. 关系标准化结果缓存
2. 关系等价性判断缓存
3. 向量嵌入缓存
4. 本地状态持久化

### 预筛选策略

实体合并时：
1. 按类型分组，只比较同类型实体
2. 名称长度差异过大的实体跳过

## 扩展开发

### 添加新的提示词

1. 在 `prompts/` 目录创建新的 `.txt` 文件
2. 在 `settings.yaml` 中添加配置
3. 在代码中通过 `qa_engine.prompts["your_prompt"]` 访问

### 自定义查询模板

编辑 `templates/query_templates.yaml`，添加新的 Cypher 查询模板。

### 扩展代理

继承 `BaseAgent` 类实现自定义代理：

```python
from aurora_core import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="custom", description="自定义代理")
    
    def execute(self, context, **kwargs):
        # 实现自定义逻辑
        return result
```

## 技术栈

- **后端**: Python 3.10+, FastAPI
- **图数据库**: Neo4j 5.0+
- **向量模型**: sentence-transformers
- **LLM**: DeepSeek API (可替换为其他 OpenAI 兼容 API)
- **前端**: React 18, Vite, TypeScript
- **容器化**: Docker, Docker Compose

## 许可证

MIT License

