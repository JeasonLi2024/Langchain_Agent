# LangChain Agent 智能体后端开发与集成指南

## 1. 简介
本文档旨在为前端开发者和运维人员提供关于 LangChain 智能体后端的部署、监管及接口调用指南。后端基于 `LangGraph` 和 `FastAPI` 构建，支持流式输出 (SSE)、多模态交互和**会话持久化**。

---

## 2. 部署与监管 (Supervisor)
生产环境中，建议使用 Supervisor 来管理 Python 进程，确保服务在意外退出时自动重启。

### 2.1 推荐配置 (`supervisord.conf`)
```ini
[program:langchain-server]
; 项目根目录 (根据实际路径调整)
directory=/mnt/data/langchain-v2.0
; 启动命令 (使用 Gunicorn 多 Worker 模式)
command=/home/bupt/Server_Project_ZH/venv/bin/gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:50018 --timeout 120
; 进程管理
autostart=true
autorestart=true
startsecs=5
; 日志配置 (带日志轮转)
stderr_logfile=/mnt/data/langchain-v2.0/logs/server.err.log
stdout_logfile=/mnt/data/langchain-v2.0/logs/server.out.log
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=10
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
; 环境变量
environment=PORT=50018,HOST="0.0.0.0",PYTHONUNBUFFERED=1,DJANGO_PROJECT_ROOT="/home/bupt/Server_Project_ZH"

[program:langchain-cleanup]
; 独立清理服务
command=/home/bupt/Server_Project_ZH/venv/bin/python scripts/cleanup_task.py
autostart=true
autorestart=true
```

### 2.2 Redis 配置说明
智能体使用 Redis 来存储对话历史（Checkpointer），实现会话持久化。**Redis 是服务器上的本地服务**（通常与应用部署在同一台机器或内网）。

**配置位置**：项目根目录下的 `.env` 文件。

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

如果需要修改 Redis 连接（例如连接到远程 Redis），请修改 `.env` 文件中的上述变量，并重启服务。

---

## 3. 前端对接指南 (REST API)

### 3.1 核心机制：REST API 与 状态管理
由于后端已配置 **Redis 持久化存储** (Checkpointer)，**前端无需回传完整历史记录**。
后端会根据 `thread_id` 自动加载上下文。
前端只需发送**当前用户消息**，并保持 `thread_id` 一致即可。

*   **对话历史 (`messages`)**：前端仅需维护用于展示的列表。每次请求**只发送最新的用户消息**，后端会自动追加到 Redis 历史中。
*   **状态字段 (`user_profile`)**：由后端自动持久化。前端**无需强制回传**，但建议在本地更新以保持 UI 状态同步。

为了实现**多用户独立对话**和**页面刷新后保持历史**，前端必须在请求中传递 `thread_id`。
*   **会话隔离**：通过 `thread_id` 区分不同用户的对话。推荐使用 `user_<uid>` 格式。
*   **持久化**：后端利用 Redis 自动保存对话状态。只要 `thread_id` 不变，历史记录就会保留。

### 3.2 接口通用规范
*   **基础 URL**: `http://<server_ip>:50018`
*   **请求方式**: `POST`
*   **响应模式**:
    *   **推荐**: 使用 `/stream_events` 后缀获取详细事件流（包含 LLM 的逐字输出），以实现流畅的“打字机效果”。
    *   备选: 使用 `/stream` 后缀获取状态更新流（粒度较粗，通常是一整段话）。

### 3.3 关于流式输出模式 (Stream Modes)
LangGraph 支持多种流式模式，不同的模式会产生不同的 SSE 事件类型：

1.  **Values Mode (`stream_mode="values"`)**:
    *   默认模式（使用 `/stream` 且不指定参数时）。
    *   每次图状态更新时，发送完整的状态对象。
    *   **不适合**实现打字机效果，因为是一次性吐出整段话。

2.  **Messages Mode (`stream_mode="messages"`)**:
    *   如果客户端请求了此模式（通常是 LangGraph JS/Python Client 的默认行为）。
    *   **`messages/partial`**: 代表 LLM 正在生成的增量 Token（打字机效果）。包含 `delta` 字段。
    *   **`messages/complete`**: 代表一个节点执行完毕，生成了完整的消息对象。
    *   **前端处理建议**: 
        *   收到 `messages/partial` -> 累加内容到当前气泡。
        *   收到 `messages/complete` -> 结束当前气泡，准备接收下一条或结束。

3.  **Updates Mode (`stream_mode="updates"`)**:
    *   仅发送状态的增量更新（例如某个节点只更新了 `messages` 字段）。

4.  **Events Mode (`/stream_events` 接口)** (本文档推荐):
    *   最底层的详细事件流。
    *   包含 `on_chat_model_stream` (打字机), `on_chain_start`, `on_chain_end` 等。
    *   **优点**: 能获取最细粒度的控制（如区分“思考过程”节点和“最终回复”节点）。

### 3.4 数据流转生命周期 (Interaction Lifecycle)
为了更好地理解无状态交互，请参考以下“请求-响应”循环：

1.  **初始请求 (T0)**:
    *   **前端发送**: `messages=[你好]`, `user_info={id:1}`
    *   **后端处理**: 智能体进行第一次思考。
    *   **后端返回**: `messages=[你好！]`, `user_profile=NULL` (或空对象)

2.  **第二轮请求 (T1)**:
    *   **前端发送**: `messages=[你好, 你好！, 我想要Python项目]`, `user_info={id:1}`
    *   **后端处理**: 智能体分析出用户喜欢Python，生成推荐。
    *   **后端返回**: 
        *   `messages=[为您推荐以下Python项目...]`
        *   `user_profile={interest_tags: [{name: Python}], recommended_projects: [{id: 101}]}`

3.  **第三轮请求 (T2)**:
    *   **前端发送**: 
        *   `messages=[请问第一个项目多少钱？]` (仅发送最新消息)
        *   `user_info={id:1}`
        *   `config={configurable: {thread_id: ...}}`
        *   `user_profile={...}` (可选：前端可省略，后端自动从 Redis 读取)
    *   **后端处理**: 智能体根据 `thread_id` 从 Redis 加载 `user_profile`，知道“第一个项目”是指 ID 101，进入问答流程。
    *   **后端返回**: 
        *   `messages=[ID 101 的预算是 50万...]`
        *   `user_profile={...}` (继续保持或更新)

**总结**: 前端只需关注 UI 展示和维护 `thread_id`，复杂的状态管理和历史记录由后端 Redis 自动处理。前端变得更轻量、更简单。

---

## 4. 智能体接口详情与 Apifox 示例

### 4.1 学生/主智能体 (Student Agent)
*   **路由**: `/student/stream_events`
*   **功能**: 闲聊、项目推荐、项目问答。
*   **参数详情 (Input Schema)**:

| 参数字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `messages` | List | **是** | **仅包含最新一条用户消息**。后端自动从 Redis 加载历史。 |
| `user_info` | Dict | **是** | 用户身份信息。 |
| `user_info.id` | Int | **是** | 学生ID (User Model ID)。 |
| `user_info.name` | Str | 否 | 学生姓名。 |
| `config` | Dict | **是** | **配置参数**。必须包含 `configurable: { thread_id: "..." }` 以启用持久化记忆（thread_id: user_${userInfo.id} ：这是实现多用户隔离的唯一关键点）。 |
| `user_profile` | Dict | 否 | **(可选) 推荐上下文记忆**。后端已自动持久化，前端无需强制回传，但可用于本地状态同步。 |
| `user_profile.interest_tags` | List[Dict] | 否 | 兴趣标签列表。例如 `[{"id": 1, "name": "Web开发"}]`。 |
| `user_profile.recommended_projects` | List[Dict] | 否 | 推荐项目列表。用于问答时定位项目（如“第一个项目”对应哪个ID）。 |

*   **关于 `user_profile`**:
    *   **持久化**: 后端会自动保存用户的兴趣和推荐结果到 Redis。
    *   **上下文定位**: 只要 `thread_id` 一致，后端就能通过历史记录找到对应的 `interest_tags` 和 `recommended_projects`。

*   **Apifox 请求示例 (Body: JSON)**:

#### 场景 A: 初始对话
```json
{
    "input": {
        "messages": [
            {"role": "user", "content": "你好，我想找一个Python项目"}
        ],
        "user_info": {
            "id": 123,
            "name": "张三"
        }
    },
    "config": {
        "configurable": {
            "thread_id": "user_123"
        }
    }
}
```

#### 场景 B: 对话进行中 (利用 Redis 记忆)
```json
{
    "input": {
        "messages": [
            // 仅发送最新消息，后端会自动拼接历史
            {"role": "user", "content": "请详细介绍一下第一个项目"}
        ],
        "user_info": {
            "id": 123
        },
        // [Optional] 可选：回传状态作为容灾，或完全省略依赖后端 Redis
        "user_profile": {
            "interest_tags": [
                {"id": 10, "name": "Web开发", "score": 0.9}
            ],
            "recommended_projects": [
                {"id": 101, "title": "企业级知识库系统", "status": "in_progress"}
            ] 
        }
    },
    "config": {
        "configurable": {
            "thread_id": "user_123"
        }
    }
}
```

### 4.2 发布者智能体 (Publisher Agent)
*   **路由**: `/publisher/stream_events`
*   **功能**: 企业需求发布、文件解析、标签推荐。
*   **参数详情 (Input Schema)**:

| 参数字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `messages` | List | **是** | **仅包含最新一条用户消息**。 |
| `user_info` | Dict | **是** | 用户身份信息。 |
| `user_info.id` | Int | **是** | 用户ID (User Model ID)。 |
| `user_info.org_id` | Int | **是** | 组织ID (Organization Model ID)。 |
| `user_info.role` | Str | 否 | 用户角色 (如 "enterprise")。 |
| `config` | Dict | **是** | **配置参数**。必须包含 `configurable: { thread_id: "..." }`。 |
| `publisher_state` | Dict | 否 | **(可选) 发布状态**。后端已自动持久化。 |
| `file_path` | Str | 否 | 上传文件的临时路径。 |
| `original_filename`| Str | 否 | 原始文件名。 |
| `parsed_file_data` | Dict | 否 | 文件解析后的结构化数据。 |

*   **Apifox 请求示例 (Body: JSON)**:

#### 场景 A: 初始对话 (无状态)
```json
{
    "input": {
        "messages": [
            {"role": "user", "content": "我要发布一个需求"}
        ],
        "user_info": {
            "id": 456,
            "org_id": 6,
            "role": "enterprise"
        }
    },
    "config": {
        "configurable": {
            "thread_id": "org_6_user_456"
        }
    }
}
```

#### 场景 B: 对话进行中 (利用 Redis 记忆)
```json
{
    "input": {
        "messages": [
            // 仅发送最新消息
            {"role": "user", "content": "企业知识库构建"}
        ],
        "user_info": {
            "id": 456,
            "org_id": 6
        },
        // [Optional] 可选：回传状态
        "publisher_state": {
            "current_draft_id": 0,
            "next_step": "refining"
        }
    },
    "config": {
        "configurable": {
            "thread_id": "org_6_user_456"
        }
    }
}
```

### 4.3 问答智能体 (QA Agent)
*   **路由**: `/qa/stream_events`
*   **功能**: 针对特定项目的深度问答。
*   **Apifox 请求示例**:
```json
{
    "input": {
        "messages": [
            {"role": "user", "content": "这个项目的技术难点是什么？"}
        ],
        "target_project_id": 1001,
        "user_info": {"id": 123}
    }
}
```

---

## 5. 前端特殊功能实现

### 5.1 识别“思考过程” (Thinking Process)
在需求推荐智能体中，AI 会在给出最终推荐前进行“思考”。前端可以通过以下两种方式识别并展示这一过程：

#### 方法一：通过流式事件 (stream_events) 实时识别
监听 SSE 流中的 `on_chat_model_stream` 事件，并检查 metadata：
1.  **事件源**: `event: on_chat_model_stream`
2.  **来源节点**: `metadata.langgraph_node` 为 `"reasoning_gen"`
3.  **内容特征**: 原始内容包含 `<thinking>...</thinking>` 标签和 JSON 代码块。

#### 方法二：通过消息元数据识别 (推荐)
后端已在最终生成的 `AIMessage` 中添加了 `name="reasoning"` 属性。
当收到 `event: on_chain_end` 且 `name: reasoning_gen` 时，查看输出的 `messages`：
```json
{
    "messages": [
        {
            "type": "ai",
            "content": "<thinking>我需要分析用户需求...</thinking>\n```json\n{...}\n```",
            "name": "reasoning" // 关键标识
        }
    ]
}
```
**前端逻辑 (Critical)**：
*   **后端不再预处理内容**：返回的 `content` 是 LLM 的完整原始输出。
*   **前端需自行解析**：
    *   提取 `<thinking>...</thinking>` 之间的内容作为“思考过程”展示。
    *   忽略或隐藏后续的 JSON 代码块（该部分是给后端解析器使用的）。
*   如果收到 `name="reasoning"` 的消息，渲染为折叠的“思考过程”组件。

### 5.2 前端完整代码示例 (JavaScript / Fetch)

以下代码展示了如何调用智能体 API，并实现**会话隔离**（基于用户 ID）和**流式打字机效果**。同时展示了如何解析 `updates` 模式下的数据结构。

```javascript
/**
 * 智能体对话 API 调用函数 (支持会话隔离与状态回传)
 * @param {string} userMessage - 用户输入的文本
 * @param {object} userInfo - 登录用户信息，必须包含 id (如 {id: 1001, name: "张三"})
 * @param {array} historyMessages - 前端维护的聊天记录列表
 * @param {object} lastUserProfile - 上一轮对话返回的 user_profile (如果是第一轮则为 null)
 * @param {function} onToken - 回调函数，用于接收流式输出的每个片段 (打字机效果)
 * @returns {Promise<object>} 返回最新的 user_profile 供下一轮使用
 */
async function callStudentAgent(userMessage, userInfo, historyMessages, lastUserProfile, onToken) {
    // 1. 构造请求 Payload
    // 核心机制：将 userInfo.id 映射为 thread_id，实现会话隔离
    const threadId = `user_${userInfo.id}`; 
    
    // 构造 LangGraph 所需的 input 对象
    const inputPayload = {
        // 仅发送最新一条消息 (后端会自动从 Redis 加载历史)
        messages: [{ type: "human", content: userMessage }],
        // 注入用户信息
        user_info: userInfo,
        // 可选：回传状态 (如果完全依赖 Redis，此处可省略；保留可作为容灾)
        user_profile: lastUserProfile || {}
    };

    try {
        // 2. 发起流式请求 (使用 /stream 接口)
        // 注意：使用 updates 模式获取增量更新
        const response = await fetch('http://<您的服务器IP>:50018/student/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                input: inputPayload,
                // 关键配置：会话隔离 ID
                config: {
                    configurable: {
                        thread_id: threadId
                    }
                },
                stream_mode: "updates"
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // 3. 处理 SSE 流式响应
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let finalUserProfile = lastUserProfile; // 用于存储更新后的状态

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            // LangServe 的流式数据格式通常是 "event: ... \n data: {...} \n\n"
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const dataStr = line.slice(6); // 去掉 "data: "
                        const data = JSON.parse(dataStr);
                        
                        // 检查是否有状态更新
                        // updates 模式下，data 是 {node_name: {updated_field: value}}
                        for (const nodeName in data) {
                            const nodeUpdate = data[nodeName];
                            
                            // 1. 提取消息内容 (用于打字机显示)
                            if (nodeUpdate.messages && nodeUpdate.messages.length > 0) {
                                const lastMsg = nodeUpdate.messages[nodeUpdate.messages.length - 1];
                                if (lastMsg.content) {
                                    // 识别思考过程
                                    if (lastMsg.content.includes("<thinking>")) {
                                        // 可选：在这里解析思考过程并单独展示
                                        console.log("正在思考...", lastMsg.content);
                                    }
                                    onToken(lastMsg.content); // 回调给 UI 显示
                                }
                            }
                            
                            // 2. 提取状态更新 (用于下轮回传)
                            if (nodeUpdate.user_profile) {
                                finalUserProfile = nodeUpdate.user_profile;
                            }
                        }
                    } catch (e) {
                        console.warn("解析 SSE 数据失败:", e);
                    }
                }
            }
        }
        
        return finalUserProfile;

    } catch (error) {
        console.error("对话请求失败:", error);
        throw error;
    }
}
```

### 3.4 数据流转生命周期
1.  **前端**: 生成 `thread_id` -> 发起请求。
2.  **后端**: 根据 `thread_id` 从 Redis 加载之前的对话状态 (Checkpointer)。
3.  **后端**: 执行 Graph 逻辑，生成回复。
4.  **后端**: 将最新的对话状态保存回 Redis。
5.  **前端**: 接收回复并更新 UI。
