# LangChain Agent 智能体后端开发与集成指南

## 1. 简介
本文档旨在为前端开发者和运维人员提供关于 LangChain 智能体后端的部署、监管及接口调用指南。后端基于 `LangGraph` 和 `FastAPI` 构建，支持流式输出 (SSE) 和多模态交互。

---

## 2. 部署与监管 (Supervisor)
生产环境中，建议使用 Supervisor 来管理 Python 进程，确保服务在意外退出时自动重启。

### 2.1 推荐配置 (`supervisord.conf`)
```ini
[program:langchain-server]
; 项目根目录 (根据实际路径调整)
directory=d:\Server_Project_ZH\langchain-v2.0
; 启动命令 (建议使用虚拟环境中的 python)
command=python server.py
; 进程管理
autostart=true
autorestart=true
startsecs=5
; 日志配置
stderr_logfile=d:\Server_Project_ZH\langchain-v2.0\logs\server.err.log
stdout_logfile=d:\Server_Project_ZH\langchain-v2.0\logs\server.out.log
; 环境变量 (覆盖 server.py 默认值)
environment=PORT=50018,HOST=0.0.0.0,PYTHONUNBUFFERED=1
```

---

## 3. 前端对接指南 (REST API)

### 3.1 核心机制：REST API 与 状态管理
由于后端采用 REST API 模式（无状态），**前端必须承担“记忆”和“状态回传”的职责**。

*   **对话历史 (`messages`)**：前端需要维护一个列表，每次请求时将**所有**历史消息（包括用户和AI的回复）打包发送。
*   **状态字段 (`user_profile`, `publisher_state`)**：后端在响应中返回的特定状态字段，前端必须在**下一轮请求中原样回传**，否则智能体会丢失上下文（例如忘记正在编辑的草稿）。

**问：如果自己开发前端，是否需要手动添加这些参数？**
**答：是的。** LangSmith 等调试工具会自动维护状态，但自研前端必须手动实现：
1.  **从登录态获取 `user_info`** 并注入每次请求。
2.  **监听后端响应**，提取 `user_profile` 或 `publisher_state` 等状态字段。
3.  **在下一轮请求中**将这些字段重新放回请求体中。

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
        *   `messages=[..., 我想要Python项目, 为您推荐..., 请问第一个项目多少钱？]`
        *   `user_info={id:1}`
        *   **`user_profile={interest_tags: [...], recommended_projects: [{id: 101}]}`** (前端原样回传 T1 的结果)
    *   **后端处理**: 智能体读取 `user_profile`，知道“第一个项目”是指 ID 101，进入问答流程。
    *   **后端返回**: 
        *   `messages=[ID 101 的预算是 50万...]`
        *   `user_profile={...}` (继续保持或更新)

**总结**: 前端就像一个“接力跑运动员”，必须接过后端递来的“接力棒 (状态对象)”，并在跑完一圈（用户输入）后，将“接力棒”再递回给后端。

---

## 4. 智能体接口详情与 Apifox 示例

### 4.1 学生/主智能体 (Student Agent)
*   **路由**: `/student/stream_events`
*   **功能**: 闲聊、项目推荐、项目问答。
*   **参数详情 (Input Schema)**:

| 参数字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `messages` | List | **是** | 对话历史列表。 |
| `user_info` | Dict | **是** | 用户身份信息。 |
| `user_info.id` | Int | **是** | 学生ID (User Model ID)。 |
| `user_info.name` | Str | 否 | 学生姓名。 |
| `user_profile` | Dict | **动态必填** | **推荐上下文记忆**。用于记录用户的兴趣标签和推荐结果。如果上一轮后端返回了此对象，前端必须在下一轮请求中**原样回传**。 |
| `user_profile.interest_tags` | List[Dict] | 否 | 兴趣标签列表。例如 `[{"id": 1, "name": "Web开发"}]`。 |
| `user_profile.recommended_projects` | List[Dict] | 否 | 推荐项目列表。用于问答时定位项目（如“第一个项目”对应哪个ID）。 |

*   **为什么要回传 `user_profile`?**
    *   **上下文定位**: 当用户说“我对第一个项目感兴趣”时，后端需要通过 `recommended_projects` 里的顺序找到对应的 `project_id`（如 ID: 101），从而进入问答模式。
    *   **个性化记忆**: `interest_tags` 记录了通过对话分析出的用户偏好，避免智能体反复询问“你对什么感兴趣”。

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
    }
}
```

#### 场景 B: 对话进行中 (携带推荐记忆)
```json
{
    "input": {
        "messages": [
            {"role": "user", "content": "我想找一个Python Web项目"},
            {"role": "assistant", "content": "根据您的兴趣，为您推荐以下项目..."},
            {"role": "user", "content": "请详细介绍一下第一个项目"}
        ],
        "user_info": {
            "id": 123
        },
        // [CRITICAL] 必须回传上一轮生成的 user_profile
        "user_profile": {
            "interest_tags": [
                {"id": 10, "name": "Web开发", "score": 0.9}
            ],
            "recommended_projects": [
                {"id": 101, "title": "企业级知识库系统", "status": "in_progress"}
            ] 
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
| `messages` | List | **是** | 对话历史列表。 |
| `user_info` | Dict | **是** | 用户身份信息。 |
| `user_info.id` | Int | **是** | 用户ID (User Model ID)。 |
| `user_info.org_id` | Int | **是** | 组织ID (Organization Model ID)。 |
| `user_info.role` | Str | 否 | 用户角色 (如 "enterprise")，逻辑中暂未使用，建议保留。 |
| `publisher_state` | Dict | **动态必填** | **关键状态对象**。如果上一轮响应中包含此字段，**必须**在下一轮请求中原样回传，否则会丢失草稿进度。 |
| `file_path` | Str | 否 | 上传文件的临时路径 (通常由第一轮 `router` 处理后生成)。 |
| `original_filename`| Str | 否 | 原始文件名。 |
| `parsed_file_data` | Dict | 否 | 文件解析后的结构化数据 (如果涉及文件解析流程)。 |

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
    }
}
```

#### 场景 B: 对话进行中 (携带状态)
```json
{
    "input": {
        "messages": [
            {"role": "user", "content": "我要发布一个需求"},
            {"role": "assistant", "content": "好的，请问项目标题是什么？"},
            {"role": "user", "content": "企业知识库构建"}
        ],
        "user_info": {
            "id": 456,
            "org_id": 6
        },
        // [CRITICAL] 必须回传上一轮后端返回的 publisher_state
        "publisher_state": {
            "current_draft_id": 0,
            "is_complete": false,
            "next_step": "refining",
            "draft_data": {
                "title": "企业知识库构建", 
                "description": "",
                "budget": "50"
            },
            "suggested_tags": {},
            "selected_tags": {}
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

### 5.2 前端伪代码示例 (JavaScript / EventSource)
```javascript
async function sendMessage(userText) {
  // 1. 更新本地消息列表
  chatHistory.push({ role: "user", content: userText });

  // 2. 准备请求体
  const payload = {
    input: {
      messages: chatHistory,
      user_info: getCurrentUser(),
      publisher_state: lastAgentState // 回传状态
    }
  };

  // 3. 发起请求
  const response = await fetch("/student/stream_events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split("\n\n");
      
      for (const line of lines) {
          if (line.startsWith("event: data")) {
             const data = JSON.parse(line.replace("data: ", ""));
             
             // --- 实时流处理 ---
             if (data.event === "on_chat_model_stream") {
                 const token = data.data.chunk.content;
                 const nodeName = data.metadata.langgraph_node;
                 
                 if (nodeName === "reasoning_gen") {
                     // 这是一个思考过程的 token
                     appendToThinkingUI(token); 
                 } else {
                     // 普通对话 token
                     appendToChatUI(token);
                 }
             }
          }
      }
  }
}
```
