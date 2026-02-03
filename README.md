# Enterprise LangChain Agent System

## 项目简介
本项目基于 **LangChain v0.2** 和 **LangGraph** 框架，构建了两个面向企业级场景的智能体（Agent）：
1. **学生智能体 (Student Agent)**：提供智能项目推荐与咨询问答服务。
2. **发布者智能体 (Publisher Agent)**：协助企业用户高效发布项目需求，支持多模态（文件/对话）信息录入。

项目采用 **FastAPI + LangServe** 部署，结合 **MySQL** (元数据) 和 **Milvus** (向量数据) 实现混合检索与持久化存储。

---

## 项目文件树及作用

```bash
/mnt/data/langchain-v2.0/
├── core/                       # 核心基础设施层
│   ├── config.py               # 全局配置（LLM/DB连接/API Keys）
│   ├── django_setup.py         # Django 环境初始化（ORM支持）
│   ├── embedding_service.py    # 向量化服务（DashScope Embeddings + Milvus）
│   └── prompts.py              # Prompt 集中管理仓库
├── graph/                      # LangGraph 智能体图定义
│   ├── main_agent.py           # [学生智能体] 主路由与RAG问答逻辑
│   ├── student_workflow.py     # [学生智能体] 推荐引擎子图（多路召回+重排序）
│   ├── publisher_main_agent.py # [发布者智能体] 主路由与多模态入口
│   ├── publisher_agent.py      # [发布者智能体] 核心发布流程状态机
│   ├── file_parsing_graph.py   # [发布者智能体] 文件解析子图
│   └── tag_recommendation.py   # 标签推荐逻辑节点
├── tools/                      # 工具集（Tools）
│   ├── new_search_tools.py     # 高级检索工具（标签/语义/关键词多路召回）
│   ├── search_tools.py         # 基础搜索与关键词提取工具
│   └── db_tools.py             # 数据库操作工具
├── scripts/                    # 运维与数据脚本
│   ├── vectorize_projects.py   # 项目数据向量化脚本
│   └── vectorize_tags.py       # 标签数据向量化脚本
├── server.py                   # FastAPI 服务启动入口
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明文档
```

---

## 智能体详细设计

### 1. 学生智能体 (Student Agent)
**定位**：为学生/开发者提供个性化的企业项目推荐及深度咨询。

#### 核心架构
*   **Router (路由节点)**：基于用户意图（RECOMMEND / PROJECT_QA / CHAT）分发请求。
*   **推荐引擎 (Recommendation Engine)**：
    *   **多路召回 (Multi-track Recall)**：
        1.  **标签召回 (Tag-based)**：基于 Interest/Skill ID 的精确匹配 (MySQL)。
        2.  **语义召回 (Semantic)**：基于向量相似度的模糊匹配 (Milvus `project_embeddings`)。
        3.  **关键词召回 (Keyword)**：基于全文索引的字面匹配 (MySQL `MATCH AGAINST`)。
    *   **重排序 (Rerank)**：加权融合三路得分，并基于内容指纹去重。
    *   **推理生成 (Reasoning)**：LLM 根据用户画像与候选列表，生成 Top 5 推荐理由（Thinking + JSON 格式）。
*   **项目咨询 (Project QA)**：
    *   采用 **RAG (检索增强生成)** 技术。
    *   **Retrieve**: 调用 `retrieve_project_chunks` 从 Milvus `project_raw_docs` 获取项目详细文档片段。
    *   **Generate**: 将检索到的上下文注入 Prompt，回答用户关于特定项目的细节问题。

### 2. 发布者智能体 (Publisher Agent)
**定位**：辅助企业用户通过对话或文件快速发布标准化的项目需求。

#### 核心架构
*   **多模态路由 (Multimodal Router)**：
    *   识别用户是否上传文件（PDF/Word）。
    *   若上传文件，路由至 **文件解析子图**；若为纯文本，路由至 **发布流程**。
*   **文件解析子图 (File Parsing Subgraph)**：
    *   `Loader` -> `Cleaner` -> `Ranking` (筛选关键片段) -> `Extractor` (LLM 结构化提取 8 大核心字段)。
*   **发布流程状态机 (Publisher Flow)**：
    *   **信息收集**：通过多轮对话补全 8 个核心字段（标题、简介、描述、技术栈等）。
    *   **标签推荐**：基于项目描述，调用 LLM 推荐 3 个兴趣标签 + 5 个技能标签。
    *   **发布决策**：用户确认预览信息后，调用 `save_requirement` 工具写入数据库（支持 Draft/Under Review 状态）。

---

## 参考文档 (LangChain Official)

本项目的设计与实现深度参考了以下官方文档与最佳实践：

1.  **LangGraph State Management**: [Dynamic Runtime Context](https://python.langchain.com/docs/concepts/#context)
    *   使用 `StateGraph` 管理对话历史、用户画像及推荐列表等短期记忆。
2.  **RAG Implementation**: [Build a custom RAG agent](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
    *   实现了标准的 Retrieve-Augment-Generate 流程。
3.  **Context Engineering**: [Prompt Injection & Short-term Memory](https://python.langchain.com/docs/concepts/#memory)
    *   通过 System Prompt 注入推荐历史，实现带上下文的连续对话。
4.  **LangServe Deployment**: [LangServe Overview](https://python.langchain.com/docs/langserve/)
    *   使用 `add_routes` 快速暴露 REST API。

---

## 服务启动与调用指南（仍在开发，此处为AI生成的预示例，目前可通过Langchain开源的Agent_Chat_UI对话体验，详见下文）

### 1. 后端服务启动

确保已配置好 `core/config.py` 中的数据库与 LLM 密钥，然后运行：

```bash
# 在 /mnt/data/langchain-v2.0 目录下
source /home/bupt/Server_Project_ZH/venv/bin/activate
python server.py
```
*   服务默认运行在 `http://0.0.0.0:50018`
*   API 文档地址: `http://localhost:50018/docs`

### 2. 前端调用接口指南

后端基于 **LangServe** 标准协议，前端可通过 HTTP POST 请求与智能体交互。

#### A. 调用学生智能体 (Student Agent)
*   **Endpoint**: `/student/invoke`
*   **Method**: `POST`

**请求示例 (Request):**
```json
{
    "input": {
        "messages": [
            {
                "role": "user",
                "content": "我想找一个关于Python的Web开发项目"
            }
        ],
        "user_info": {"id": 123}  // 可选：传递用户ID
    },
    "config": {},
    "kwargs": {}
}
```

**响应示例 (Response):**
```json
{
    "output": {
        "messages": [
            {
                "content": "根据您的需求，为您推荐以下项目...",
                "type": "ai"
            }
        ],
        "user_profile": { ... } // 返回更新后的用户画像
    }
}
```

#### B. 调用发布者智能体 (Publisher Agent)
*   **Endpoint**: `/publisher/invoke`
*   **Method**: `POST`

**请求示例 (Request):**
```json
{
    "input": {
        "messages": [
            {
                "role": "user",
                "content": "我要发布一个新需求"
            }
        ],
        "user_info": {"id": 557, "org_id": 6} // 必须：传递企业用户ID和组织ID
    }
}
```

### 3. 流式调用 (Streaming)
为了获得更好的打字机效果，推荐使用 `/stream` 端点：
*   **URL**: `/student/stream` 或 `/publisher/stream`
*   **处理方式**：前端需处理 Server-Sent Events (SSE)，解析 `event: data` 中的 JSON 数据块。

## 使用Langchain开源的Agent_Chat_UI对话体验，LangSmith监控可视化流程
*   **Langchain Agent_Chat_UI**: [Agent_Chat_UI](https://github.com/langchain-ai/agent-chat-ui)
    *   Agent_Chat_UI 是一个Next.js应用程序，提供与任何 LangChain 代理交互的对话接口。提供 Web 界面，支持与智能体交互、查看对话历史。
*   **LangSmith**: [LangSmith](https://www.langchain.com/langsmith)
    *   用于监控、调试与可视化 LangGraph 流程，帮助开发者理解智能体行为。

### 1. 配置LangSmith
*   注册并登录 [LangSmith](https://www.langchain.com/langsmith)。
*   在项目设置中获取 API Key。
*   在 `.env` 文件中配置 `LANGSMITH_API_KEY`。
*   确保 `LANGCHAIN_TRACING_V2` 已设置为 `true`。

### 2. 监控与调试
*   启动LangGraph Server后端服务:

```bash
# 在 /mnt/data/langchain-v2.0 目录下
source /home/bupt/Server_Project_ZH/venv/bin/activate
# --host 0.0.0.0 允许所有 IP 访问(确保可在个人电脑上本地访问)
langgraph dev --host 0.0.0.0
```
*   访问LangSmith Studio (Web UI) 连接此远程服务器。
    *   在左侧选择studio,右侧选择“Configure connection”，在“Base URL”中输入服务器的公网 IP 或域名（`http://10.3.120.200:2024`）即可查看可视化过程。
*   与智能体交互时，LangSmith 会记录每轮对话、状态更新及工具调用。

### 3. 启动Agent_Chat_UI
*   启动 Agent_Chat_UI：
```bash
# 在 /mnt/data/agent-chat-ui 目录下
pnpm dev
```
*   访问`http://10.3.120.200:3000` 进入Agent_Chat_UI界面。
*   输入“Deployment URL”为： `http://10.3.120.200:2024`,"Graph ID"从文件langgraph.json中的`graphs`字段选择(如`main_agent`或`publisher_agent`)并输入 LangSmith API key（从LangSmith Studio获取，参考`https://docs.langchain.com/langsmith/create-account-api-key`）即可连接到 LangGraph Server。
*   与智能体对话时，Agent_Chat_UI 会显示智能体的回复，同时在 LangSmith Studio 中查看详细的流程可视化。


## 关于Milvus
*   项目中使用Milvus作为向量数据库，存储和检索项目需求的向量表示以及标签的向量化匹配。
*   服务器上的Milvus使用Docker Compose配置，可输入指令查看容器状态：

```bash
docker-compose ps -a
```
    *   确保Milvus容器状态为`Up`，表示正常运行。若容器状态为`Exited`，可输入指令重启容器：

```bash
docker-compose up -d
```
*   可在Milvus正常运行期间访问Attu(Web UI) 查看Milvus数据库状态和管理向量数据。
    *   访问地址： `http://10.3.120.200:8000`
    *   输入地址后连接即可，进入后可以看到有4个collections，分别是`project_embeddings`、`project_raw_docs`、`student_interests`、`student_skills`。
