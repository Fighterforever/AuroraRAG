# AuroraRAG

AuroraRAG 是一个基于知识图谱的智能问答系统。

## 主要特性

- 知识图谱构建：自动从文档中抽取实体、关系、属性和事件
- 语义理解：基于大语言模型的实体合并、关系标准化
- 社区检测：自动识别知识社区，生成社区画像
- 智能问答：多策略混合检索，支持图路径、向量和Cypher查询
- 分布式存储：Neo4j 图数据库 + 本地向量缓存

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
```bash
docker compose -f docker/docker-compose.yml up -d
```

3. 配置系统
编辑 `settings.yaml`，配置 API 密钥和数据库连接信息。

4. 摄取文档
```bash
python main.py ingest
```

5. 开始问答
```bash
python main.py qa
```

## 技术栈

- **后端**: Python 3.10+, FastAPI
- **图数据库**: Neo4j 5.0+
- **向量模型**: sentence-transformers
- **LLM**: DeepSeek API
- **前端**: React 18, Vite, TypeScript

## 许可证

MIT License
