# 需求发布流程增强设计方案

## 1. 背景与目标

当前发布流程已经具备：
- 多轮对话发布
- 文件上传解析
- 标签推荐
- 封面图生成与选择
- 保存/发布落库
- 发布后向量同步

但还缺少两类关键能力：
- 保存工具层的字段完整性强校验，避免不完整需求被直接发布
- 发布流程中的敏感信息审核与状态回滚，避免非法内容进入数据库、向量库和附件链路

本方案目标是：
1. 当字段缺失时，强制降级为 draft，并告诉智能体缺失项，由智能体继续引导用户补齐。
2. 当检测到敏感信息时，立即中断发布，不落库，不向量化，不保留非法发布态。
3. 命中敏感信息后，回退到入口状态，并清空包含非法信息的发布相关 state。
4. 保持普通聊天上下文不受影响，避免影响非发布业务。

## 2. 适用范围

相关代码位置：
- 发布主编排图: `graph/publisher_main_agent.py`
- 发布子图: `graph/publisher_agent.py`
- 文件解析子图: `graph/file_parsing_graph.py`
- 提示词和发布规则: `core/prompts.py`

## 3. 当前流程概览

### 3.1 主编排图

发布主图入口在 `graph/publisher_main_agent.py`。
其核心节点包括：
- `router_node`: 负责识别文件上传、发布会话、闲聊
- `file_parsing_node`: 负责上传文件解析
- `publisher_bridge_node`: 负责把解析数据和对话状态传入发布子图
- `vector_sync_node`: 负责发布完成后的收尾提示

### 3.2 发布子图

发布子图入口在 `graph/publisher_agent.py`。
其核心节点包括：
- `chat_node`: 负责和用户多轮收集信息
- `tag_recommendation_node`: 负责标签推荐
- `cover_flow_node`: 负责封面图生成和选择
- `save_requirement`: 负责保存/发布需求

### 3.3 文件解析链路

文件解析子图在 `graph/file_parsing_graph.py`，负责：
- 加载文件
- 清洗与分段
- 结构化提取
- 返回解析结果供发布子图使用

## 4. 新增字段后的业务规则

保存前必须覆盖 12 个核心字段：
- 标题 title
- 简介 brief
- 详细描述 description
- 研究方向 research_direction
- 技术栈 skill
- 目标 goal
- 期望成果 expected_result
- 联系人 contact_person
- 联系方式 contact_info
- 完成时间 finish_time
- 预算 budget
- 可提供的支持 support_provided

## 5. 功能设计

### 5.1 字段完整性强校验

#### 目标
在保存工具中识别字段是否缺失，并根据发布意图决定是否允许继续。

#### 规则
1. 如果用户当前请求是 `under_review`，但字段不完整：
- 不允许继续按 `under_review` 保存
- 自动改为 `draft`
- 返回缺失字段列表
- 智能体需要告诉用户缺失项，并引导用户补齐后再发布

2. 如果用户当前请求是 `draft`：
- 可允许保存
- 仍然返回缺失字段信息，方便智能体继续补齐

#### 建议返回协议
```json
{
  "status": "warning",
  "code": "MISSING_REQUIRED_FIELDS",
  "forced_status": "draft",
  "missing_fields": ["contact_info", "expected_result"],
  "message": "字段不完整，已为你暂存草稿，请补齐后再发布"
}
```

### 5.2 敏感信息审核

#### 目标
确保上传文件和需求详情字段中不会出现非法敏感信息。

#### 审核对象
1. 文件解析结果：
- summary
- chunks
- filtered_chunks
- extracted_data

2. 需求详情字段：
- title
- brief
- description
- research_direction
- skill
- goal
- expected_result
- contact_person
- contact_info
- support_provided

#### 审核方式
采用混合模式：
1. 本地词库初筛
2. 第三方审核 API 复核

#### 判定原则
- 任一审核引擎判定为 block，则立即中断发布
- 命中敏感信息后不得落库，不得向量化，不得保留非法发布态

#### 建议返回协议
```json
{
  "status": "error",
  "code": "SENSITIVE_CONTENT_BLOCKED",
  "block": true,
  "hits": ["敏感词1", "敏感词2"],
  "stage": "file_parsing",
  "message": "检测到敏感词汇信息，请修改后再发布"
}
```

### 5.3 状态回滚

#### 目标
命中敏感信息后，立即清空发布相关 state 并回退到入口状态。

#### 清空范围
保留：
- messages
- user_info

清空：
- publisher_state
- parsed_file_data
- file_path
- original_filename
- cover_image_path
- final_requirement_id
- final_requirement_data

#### 目的
避免用户后续补发时继承非法信息残留。

## 6. 推荐拦截点

### 6.1 文件解析后拦截

位置：`graph/publisher_main_agent.py` 的 `file_parsing_node`

作用：
- 文件解析完成后立即审核提取内容
- 命中敏感信息后，直接返回错误提示
- 清空发布态，不进入发布子图

### 6.2 保存工具入口拦截

位置：`graph/publisher_agent.py` 的 `save_requirement`

作用：
- 做字段完整性校验
- 做敏感信息审核兜底
- 保证最终落库前最后一道防线

### 6.3 桥接节点拦截

位置：`graph/publisher_main_agent.py` 的 `publisher_bridge_node`

作用：
- 解析工具返回的结构化状态码
- 根据状态码决定是否进入向量同步
- 根据错误类型决定是否清空发布相关 state

## 7. 流程设计

### 7.1 正常发布流程
1. 用户进入发布入口
2. 主路由判断是发布意图
3. 如果上传文件，先走文件解析
4. 文件解析结果进入发布子图
5. 发布子图收集字段、推荐标签、封面图
6. 用户确认后调用保存工具
7. 保存工具执行字段校验与敏感信息审核
8. 校验通过则保存并返回成功
9. 主图完成附件处理与向量同步
10. 流程结束

### 7.2 缺字段流程
1. 用户请求发布
2. 保存工具发现字段缺失
3. 工具把状态强制降级为 draft
4. 返回缺失字段列表
5. 智能体提示用户补齐字段
6. 用户补齐后可再次发起发布

### 7.3 敏感信息流程
1. 用户上传文件或输入详情字段
2. 解析或保存前审核发现敏感内容
3. 立即中断发布
4. 返回敏感词检测提示
5. 清空发布相关 state
6. 回到入口状态
7. 用户修改后重新开始

## 8. 智能体行为规范

### 8.1 缺字段时
智能体必须：
- 告知用户缺失字段
- 明确说明已自动保存为草稿
- 引导用户补齐后再发布

### 8.2 敏感信息时
智能体必须：
- 告知用户检测到敏感词汇信息
- 明确说明已中断发布
- 让用户修改后重新提交

### 8.3 不要做的事
- 不要把敏感内容继续传给保存工具
- 不要在敏感拦截后继续向量同步
- 不要在状态未清空时复用旧 draft_data

## 9. 审计与日志

建议记录以下信息：
- thread_id
- user_id
- org_id
- stage
- code
- hits_count
- requirement_id（如有）
- duration_ms

注意：
- 不记录完整敏感原文
- 仅记录命中词摘要和必要的审计信息

## 10. 测试场景

### 10.1 字段缺失
- 输入 under_review，但缺 contact_info
- 预期：强制改为 draft，并提示缺失字段

### 10.2 文件含敏感词
- 上传文件解析结果包含违规内容
- 预期：立即中断，清空发布态，不落库

### 10.3 字段含敏感词
- goal、description 或 contact_info 中含违规内容
- 预期：保存工具拒绝，不落库

### 10.4 敏感拦截后再次发起发布
- 预期：从入口重新开始，不继承上次非法 state

### 10.5 正常完整发布
- 预期：正常发布、向量同步成功

## 11. 实施顺序建议

1. 先定义统一返回协议和错误码
2. 在保存工具中加入字段完整性校验和敏感审核兜底
3. 在文件解析节点加入解析后敏感审核
4. 在桥接节点加入状态回滚逻辑
5. 更新提示词中的错误码行为规范
6. 增加自动化测试和审计日志

## 12. 结论

这套方案可以实现：
- 缺字段时自动降级为 draft
- 敏感信息立即拦截
- 回退入口状态并清空发布相关 state
- 保持非发布上下文不受影响

如果后续要进入实现阶段，建议先做：
- 统一审核/校验返回协议
- 保存工具兜底校验
- 文件解析后的前置拦截
