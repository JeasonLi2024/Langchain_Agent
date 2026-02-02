# 前端流式输出实现指南 (Frontend Streaming Guide)

本文档详细说明了如何基于 LangChain 的 `astream_events` 接口，在前端实现流畅的打字机效果（Streaming Output），并正确处理和展示多智能体协作过程中的中间状态（如思考过程、关键词提取等）。

## 1. 核心原理

后端不直接返回完整的 JSON 响应，而是使用 Server-Sent Events (SSE) 或 WebSocket 推送事件流。LangChain v0.2 提供了 `astream_events` API，能够细粒度地广播工作流内部发生的各类事件。

### 关键事件类型 (Event Types)

| 事件名称 (Event Name) | 触发时机 | 前端处理逻辑 |
| :--- | :--- | :--- |
| `on_chat_model_stream` | LLM 生成新的 Token 时 | **核心**：将 Token 追加到当前对话气泡中，实现打字机效果。 |
| `on_chain_start` | 进入新的节点或 Chain 时 | **状态提示**：展示“正在分析...”、“正在思考...”等加载状态。 |
| `on_tool_start` | 开始调用工具时 | **工具日志**：(可选) 展示“正在检索数据库...”。 |
| `on_chain_end` | 节点或 Chain 执行结束时 | **清理状态**：移除加载动画，或将思考过程折叠。 |

---

## 2. 后端接口示例 (Python/FastAPI)

后端需要暴露一个支持 SSE 的接口。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from graph.main_agent import master_app  # 你的 LangGraph 应用实例
import json

app = FastAPI()

async def event_generator(user_input: str):
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # 使用 astream_events 监听所有事件
    async for event in master_app.astream_events(initial_state, version="v1"):
        kind = event["event"]
        
        # 1. 捕获 LLM 的流式输出 (Token)
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # 构造 SSE 数据格式
                payload = json.dumps({"type": "token", "content": content})
                yield f"data: {payload}\n\n"
        
        # 2. 捕获节点状态变化 (Status)
        elif kind == "on_chain_start":
            node_name = event["name"]
            if node_name in ["extract_keywords", "reasoning", "recommendation_flow"]:
                payload = json.dumps({"type": "status", "node": node_name})
                yield f"data: {payload}\n\n"

@app.get("/chat/stream")
async def chat_stream(query: str):
    return StreamingResponse(event_generator(query), media_type="text/event-stream")
```

---

## 3. 前端实现逻辑 (伪代码/React)

前端需要建立 SSE 连接，并根据接收到的事件类型更新 UI。

### 3.1 建立连接

```javascript
const eventSource = new EventSource(`/chat/stream?query=${encodeURIComponent(userInput)}`);

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleEvent(data);
};
```

### 3.2 处理事件 (Handle Event)

我们需要维护两个主要的 UI 区域：
1. **思考过程/系统日志 (Thinking/System)**：用于展示中间步骤。
2. **最终回复 (Final Response)**：用于展示 Agent A 的对话内容。

```javascript
let currentMode = "response"; // 'thinking' | 'response'

function handleEvent(data) {
    if (data.type === "status") {
        // --- 处理状态切换 ---
        if (data.node === "reasoning") {
            currentMode = "thinking";
            showThinkingIndicator("正在深度思考...");
        } else if (data.node === "extract_keywords") {
            showToast("正在提取关键词...");
        } else {
            // 其他节点通常意味着切回主对话
            currentMode = "response"; 
        }
    } 
    else if (data.type === "token") {
        // --- 处理文本流 ---
        if (currentMode === "thinking") {
            // 将 Token 追加到“思考过程”折叠面板中
            appendToThinkingBlock(data.content);
        } else {
            // 将 Token 追加到主聊天气泡中
            appendToChatBubble(data.content);
        }
    }
}
```

## 4. 最佳实践与注意事项

### 4.1 区分“思考”与“回答”
在 `student_workflow` 中，`reasoning` 节点的输出通常是复杂的分析过程（包含 `<thinking>` 标签）。
- **建议**：前端应检测到 `<thinking>` 标签时，自动将其渲染为一个**可折叠的详情组件 (Accordion)**，默认折叠，用户点击可展开查看详细推理过程。
- **Agent A 回复**：Agent A 的回复应始终作为普通的 Markdown 文本渲染。

### 4.2 处理延迟
由于 LangGraph 需要先执行子图才能生成最终回复，中间可能会有几秒钟的“空白期”。
- **必须**：利用 `on_chain_start` 事件展示动态的 Loading 状态（如“正在搜索项目库...”、“Agent A 正在整理结果...”），避免用户以为界面卡死。

### 4.3 错误处理
记得监听 SSE 的 `onerror` 事件，以便在连接断开或后端报错时提示用户重试。

```javascript
eventSource.onerror = (err) => {
    console.error("Stream failed:", err);
    showError("连接断开，请重试");
    eventSource.close();
};
```
