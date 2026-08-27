## Plan: 学生主档读取、对话补全与推荐画像当前态联动

目标是把学生端推荐从“单轮输入 -> 三路召回 -> 精排生成”升级为“读取学生主档与标签事实 -> 多轮追问补全 -> LLM总结画像摘要 -> 标签补充 -> 推荐召回 -> 证据留存 -> 当前态持续更新”的闭环。核心原则是：`Student` 与 `Tag1StuMatch/Tag2StuMatch` 仍是权威事实层，推荐画像表只保留当前态核心结果，不再维护历史态表；会话短期状态归 LangGraph，长期当前态与证据归 MySQL，语义索引归 Milvus。

**总体设计**
- 推荐开始前，智能体必须先读取 `Student` 表和 `Tag1StuMatch/Tag2StuMatch` 表的已有信息。
- 如果 `Student` 主档或标签事实层存在关键字段缺失，智能体先以多轮问答方式补齐，再按确认结果写回数据库。
- 推荐画像当前态不再保存大量结构化槽位，只保留三个主字段：`profile_summary`、`candidate_interest_tags`、`candidate_skill_tags`。
- `profile_summary` 由 LLM 基于主档、标签、用户多轮回答统一总结生成，风格接近简历摘要，不超过 300 字。
- 每次用户对话出现新增有效信息时，都要同步更新当前 session 草稿、证据层，以及必要时更新推荐画像当前态。
- 不再单独保存画像历史态表，保留证据层用于追溯“画像是根据什么信息形成的”。

**设计目标**
- 智能体在推荐前具备“先读档、再补全、后推荐”的能力，而不是只根据当前一句话做推荐。
- 对学生来说，补问过程尽量自然，像完善简历与意向，而不是机械填表。
- 对系统来说，推荐画像当前态应尽量轻量，只保留推荐真正需要消费的核心摘要与候选标签。
- 对后端来说，避免和 `Student` 主档、`Tag1StuMatch/Tag2StuMatch` 标签事实层发生职责重叠与双写冲突。

**数据分层**
- 基础主档层：`Student` 保存学校、专业、年级、学历层次、学籍状态、认证、预计毕业时间等稳定信息。
- 标签事实层：`Tag1StuMatch`、`Tag2StuMatch` 保存学生已确认的兴趣标签与能力标签，作为权威标签来源。
- 推荐画像当前态：`student_profile_current` 仅保存当前推荐消费所需的摘要和候选标签结果。
- 画像证据层：`student_profile_evidence` 保存对话内容、人工确认、文件片段、主档读取、标签读取等证据来源。
- 语义索引层：Milvus 的 `student_profiles` 集合保存 `profile_summary` 的向量，主键直接使用 `student_id`，只做检索索引，不作为事实主库。

**推荐前必须读取的事实数据**
- 从 `Student` 中读取：学校、专业、年级、学历层次、学籍状态、认证、预计毕业时间。
- 从 `Tag1StuMatch` 中读取：已确认的兴趣标签。
- 从 `Tag2StuMatch` 中读取：已确认的能力标签。
- 若有推荐画像当前态，则同时读取已有 `profile_summary`、`candidate_interest_tags`、`candidate_skill_tags`。
- 当前 session 如已有补问进度、临时草稿、待确认字段，也一并纳入上下文。

**关键缺失项判断**
- `Student` 主档中的关键信息缺失时，优先补：学校、专业、年级、学历层次。
- 标签事实层缺失时，优先补：兴趣方向、技能栈、目标科研/就业方向。
- 若标签已有，但缺少经历与目标信息，则继续补：项目经历、竞赛经历、实践经历、目标岗位或目标研究方向。
- 若主档和标签都较完整，但仍无法形成有效推荐摘要，则至少补充一个“目标方向”与一个“经历/能力证明”。

**推荐画像当前态定义**
- `profile_summary`：一段 300 字以内的中文摘要，类似简历简介，用于描述学生当前背景、兴趣方向、技能、经历与发展目标。
- `candidate_interest_tags`：基于已确认兴趣标签、当前轮回答抽取标签、LLM推荐标签合并去重后形成的候选兴趣标签列表。
- `candidate_skill_tags`：基于已确认能力标签、当前轮回答抽取标签、LLM推荐标签合并去重后形成的候选能力标签列表。
- 当前态不再承担长期结构化事实存储职责；结构化事实仍以 `Student` 和标签关联表为准。

**profile_summary 生成要求**
- 摘要模板要接近简历简介，而不是问答复述。
- 应优先覆盖：学校、专业、学历/年级或毕业状态、兴趣方向、技能、项目/竞赛/实践经历、目标方向。
- 示例风格：该学生当前就读于 xx 大学 xx 专业，本科/硕士/博士在读，xx 年级。对 xx 方向兴趣浓厚，具备 xx 技能，曾参与 xx 项目/竞赛，希望进一步发展 xx。
- 若部分字段未知，可以省略，不要硬编造。
- 严格控制在 300 字以内，便于直接用于语义召回与推荐解释。

**Steps**
1. 在推荐入口增加 prefetch 节点，先读取 `Student`、`Tag1StuMatch`、`Tag2StuMatch`、`student_profile_current` 和当前 session 草稿。*depends on main_agent entry and student_workflow*
2. 统一主键与身份映射，确保 LangChain 能同时拿到 `user_id` 与 `student_id`，避免把用户主键误当成学生主键。*parallel with step 1*
3. 在推荐入口增加 profile gate，先判断属于“主档缺失 / 标签缺失 / 摘要不足 / 可直接推荐 / 推荐结果追问”哪一类。*depends on step 1*
4. 定义最小推荐门槛：至少具备“主档基本信息中 2 项以上 + 兴趣或技能之一 + 目标方向或经历之一”。不满足则先补问。*depends on step 3*
5. 在 `analyze_query_node` 内升级为画像补全节点，输入 `messages + user_input + Student 主档 + 已确认标签 + 当前态摘要`，抽取新增事实、候选标签和待补字段。*depends on step 4*
6. 设计多轮追问策略，一次只追问 1-2 个高价值问题，优先顺序为：学校/专业/年级/学历、兴趣方向、技能栈、项目/竞赛/实践经历、目标岗位/目标研究方向。*depends on step 5*
7. 若用户明确确认主档信息，则将补充结果写回 `Student`；若用户明确确认标签，则同步写回 `Tag1StuMatch/Tag2StuMatch`。*depends on step 6*
8. 在多轮采集达到最小门槛后，调用 LLM 对当前事实与用户回答做统一总结，生成 `profile_summary`。*depends on step 7*
9. 基于主档、已确认标签、用户回答和摘要，生成并更新 `candidate_interest_tags`、`candidate_skill_tags`。*depends on step 8*
10. 将 `profile_summary`、`candidate_interest_tags`、`candidate_skill_tags` upsert 到 `student_profile_current`。*depends on step 9*
11. 所有对话补全过程同时写入 `student_profile_evidence`，保留字段来源、原始文本、置信度与映射关系。*parallel with steps 5-10*
12. 调整三路召回输入构造。
   - 标签召回：已确认标签 + 候选标签一起参与召回，但区分 confirmed 与 candidate 权重。
   - 语义召回：`profile_summary + 当前诉求` 重写成干净 query。
   - 关键词召回：从学校、专业、技能、经历、目标方向中抽取归一化词项。
   *depends on steps 8-10*
13. 调整 rerank_node，新增与摘要和标签相关的特征：方向一致性、技能覆盖、经历相似度、标签契合度、主档匹配度。*depends on step 12*
14. 调整 reasoning_gen_node 的输出规则，推荐理由优先引用 `profile_summary` 与已确认标签；如果信息仍不完整，要明确说明当前推荐依据。*depends on steps 8-13*
15. 定义对话更新策略：每次用户回答出现新有效信息时，优先更新 session 草稿和 evidence；达到可总结条件后再刷新 `student_profile_current`。*depends on steps 5-11*
16. 定义不同 session_id 的行为边界：新 session 先读主档、标签和当前态；如仍有关键缺口，则继续补问，但不要重复询问已确认内容。*depends on steps 1, 3, 15*
17. 定义失败与回退策略：若主档读取失败则退回轻量补问；若标签写回失败则保留 session 草稿与 evidence；若摘要生成失败则用结构化事实拼接简短兜底摘要；若召回 query 重写失败则使用原始 `user_input`。*depends on steps 5, 8, 12, 15*

**数据库调整建议**
- 保留：`Student`、`Tag1StuMatch`、`Tag2StuMatch` 作为事实主库。
- 新增：`student_profile_current`，仅保存 `student_id`、`profile_summary`、`candidate_interest_tags_json`、`candidate_skill_tags_json`、`updated_at`。
- 新增：`student_profile_evidence`，保存 `student_id`、`evidence_type`、`evidence_text`、`field_mapping_json`、`evidence_confidence`、`source`、`session_id`、`created_at`。
- 删除原计划中的 `student_profile_history`，不再维护版本快照与 diff。
- Milvus 中的 `student_profiles` 仅维护当前 `profile_summary` 的向量，同步更新即可，不需要版本号主键。

**写入策略**
- `Student`：仅在用户明确提供并确认主档事实后写回，例如学校、专业、年级、学历层次。
- `Tag1StuMatch/Tag2StuMatch`：仅在兴趣/技能标签被明确确认后写回，不要把低置信度候选标签直接写成事实标签。
- `student_profile_current`：每次对话新增有效信息且能够改善摘要时进行 upsert。
- `student_profile_evidence`：高频写入，记录每轮对话和字段来源。
- Milvus：在 `profile_summary` 更新后异步重建当前向量，不阻塞主推荐链路。

**推荐入口逻辑**
- 已有完整主档、标签和可用摘要的用户：直接进入推荐主流程，最多补 1 轮确认性问题。
- 主档缺失的用户：优先补主档关键信息，再继续兴趣、技能和目标方向采集。
- 标签缺失的用户：优先引导回答兴趣方向、技能栈和目标方向，并在确认后写入标签事实层。
- 已有主档与标签但摘要不足的用户：继续问经历与目标，生成新的 `profile_summary` 后再推荐。
- 追问推荐结果的用户：不重新做全量补档，只基于已有推荐列表、`profile_summary` 和标签事实回答。
- 新 session 用户：读取长期当前态和事实层，避免重复询问已确认内容。

**Prompt 策略**
- 基础 prompt：固定角色、中文回答、推荐前先补信息的行为边界。
- 主档补全 prompt：只在 `Student` 关键字段缺失时触发，问题要简洁明确。
- 标签补全 prompt：围绕兴趣方向、技能栈、目标方向追问，支持一次回答多个点。
- 摘要生成 prompt：输入主档、标签、经历、目标和原始回答，输出 300 字以内 `profile_summary`。
- 标签推荐 prompt：基于摘要、已确认标签和当前回答，输出 `candidate_interest_tags` 与 `candidate_skill_tags`。
- 推荐 prompt：输入 `profile_summary`、已确认标签、候选标签、候选项目和用户最新诉求。
- 结果追问 prompt：针对已推荐结果解释、比较和细化，不重复触发主档补全。

**AgentState 建议**
- `student_db_profile`：读取自 `Student` 的主档信息。
- `confirmed_interest_tags`、`confirmed_skill_tags`：读取自标签事实层。
- `profile_draft`：当前 session 的临时补全结果。
- `profile_missing_fields`：仍需追问的关键字段。
- `profile_summary`：当前轮或长期当前态的摘要。
- `candidate_interest_tags`、`candidate_skill_tags`：当前推荐消费的候选标签。
- `profile_evidence`：当前 session 待写入的证据缓存。
- `profile_gate_decision`：当前入口决策，例如 `need_student_fields`、`need_tags`、`need_summary`、`recommend_ready`。

**Relevant files**
- `/home/bupt/Server_Project_ZH/user/models.py` — `Student`、`Tag1StuMatch`、`Tag2StuMatch` 以及新增 `student_profile_current/evidence` 的模型边界。
- `/home/bupt/Server_Project_ZH/user/serializers.py` — 学生主档和标签的更新逻辑，可扩展为推荐画像当前态读写。
- `/home/bupt/Server_Project_ZH/user/views.py` — 可增加内部接口或服务入口，用于智能体读取和补写主档/标签/当前态。
- `/mnt/data/langchain-v2.0/graph/main_agent.py` — 推荐入口、profile gate、session 分流。
- `/mnt/data/langchain-v2.0/graph/student_workflow.py` — 多轮补问、摘要生成、标签补充、召回、重排、推荐解释。
- `/mnt/data/langchain-v2.0/core/prompts.py` — 主档补全、标签补全、摘要生成、候选标签推荐、推荐解释 prompt。
- `/mnt/data/langchain-v2.0/tools/search_tools.py` — 从用户回答与摘要中抽取标签与关键词。
- `/mnt/data/langchain-v2.0/tools/new_search_tools.py` — 语义召回与关键词召回的输入组织。
- `/mnt/data/langchain-v2.0/tools/db_tools.py` — 读取 `Student`/标签事实层并写回主档、标签、当前态、证据层。

**Verification**
1. 用四类样例回放：主档缺失用户、标签缺失用户、摘要不足用户、已有完整画像用户，确认入口分流正确。
2. 验证智能体是否能在对话开始时成功读取 `Student` 和 `Tag1StuMatch/Tag2StuMatch` 信息。
3. 验证主档缺失项在用户回答确认后能正确写回 `Student`，标签缺失项能正确写回标签关联表。
4. 检查 `profile_summary` 是否始终控制在 300 字以内，且能覆盖学校/专业/方向/技能/经历/目标中的高价值信息。
5. 检查 `candidate_interest_tags`、`candidate_skill_tags` 是否包含“已确认标签 + 当前轮新抽取/推荐标签”的合理合并结果。
6. 对比修改前后召回结果，确认摘要驱动的语义召回更聚焦，标签召回不退化。
7. 检查推荐解释是否引用摘要和已确认标签，而不是只复述当前一句话。
8. 验证 `student_profile_current` 与 `student_profile_evidence` 能持续更新，且 Milvus 更新异步不阻塞推荐响应。

**Decisions**
- `Student` 仍是学生稳定主档，不把复杂推荐画像堆回 `Student` 表。
- `Tag1StuMatch/Tag2StuMatch` 继续作为权威标签事实层，智能体可读取，也可在用户确认后补写。
- 推荐画像当前态只保留 `profile_summary`、`candidate_interest_tags`、`candidate_skill_tags` 三项核心信息。
- 不再单独设计画像历史态表，保留证据层即可满足追溯与调试需求。
- 推荐前必须先读事实层，再决定是否进入补问，而不是直接基于单轮输入召回。
- 语义召回优先使用 `profile_summary`，标签召回同时考虑 confirmed tags 与 candidate tags，但两者权重要区分。

**Further Considerations**
1. 如果你们后续发现证据追溯还不够，再补历史态表，但首期不建议增加复杂度。
2. 如果用户不愿意回答太多问题，可以把门槛下调为“主档基本信息 + 一个方向 + 一个技能/兴趣”即可先推荐。
3. 如果推荐目标偏科研匹配，可提高专业、研究方向、项目经历的权重；若偏实习就业，可提高技能、岗位方向、实践经历权重。
4. 如果后续要接入简历上传，简历抽取结果也应优先写入 evidence，并在用户确认后同步更新 `Student`、标签事实层和当前态摘要。
5. 如果要减少重复追问，建议在 session prompt 前缀中明确列出“已确认主档信息”“已确认标签”“当前摘要”，让模型优先补缺而非重问。
