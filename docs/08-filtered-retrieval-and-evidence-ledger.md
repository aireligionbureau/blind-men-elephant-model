# 过滤式检索与证据账本

## 为什么需要过滤式检索

数字人的偏差不只发生在思考阶段，也发生在信息获取阶段。

真实的人不会平等地阅读所有信息。人会搜索自己愿意搜索的关键词，信任自己愿意信任的来源，排斥不符合价值观或专业习惯的信息，并忽略那些不容易被自己的框架命名的证据。

所以盲人摸象模型不能让数字人自由上网。每个数字人必须按自己的信息摄入身份证进入信息世界。

## 过滤式检索流程

每个数字人会生成：

- preferred_queries：偏好的搜索词
- trusted_sources：高信任来源类型
- distrusted_sources：低信任来源类型
- preferred_evidence：偏好的证据类型
- ignored_evidence：容易忽略的证据类型
- query_intent：它为什么这样搜

搜索 provider 返回候选来源后，系统不会直接喂给数字人，而是先做过滤决策：

- accept：进入该数字人的证据账本，表示“这个人愿意相信”，不自动表示“系统已核验为真”
- reject：被明确排斥，构成信息阴影
- ignore：处于低注意力区，可能构成沉默阴影

## 证据账本记录什么

证据账本不是参考文献列表，而是认知足迹。

它记录：

- 搜索策略
- 所有候选来源
- 每个来源的 accept/reject/ignore 决策
- 采信理由
- 排斥理由
- 忽略理由
- 信息阴影提示
- 被采信来源如何支撑判断
- 来源编号、原始搜索词、发布者与核验状态
- 该材料只能显影阴影、可作候选观察，还是有资格支持方向判断

`accept/reject/ignore` 描述数字人的选择，`evidence_use_role` 描述元模型允许这条材料做什么。模型先验和 mock 材料只能是 `shadow_only`；未核验搜索摘要是 `candidate_observation`；只有页面、原始来源或交叉核验材料才可能成为 `directional_eligible`，并且仍需通过来源独立性审查。

## 元模型如何使用

元模型后续不能只读数字人的观点。它必须审计：

```text
信息过滤 -> 来源决策 -> 证据账本 -> 假设形成 -> 推理路径 -> 结论输出
```

如果一个数字人的结论在自己的信息圈里高度自洽，但大量关键来源被 reject 或 ignore，那么这个结论可能是信息泡泡内的局部高清。

## 当前实现

当前实现提供：

- `SearchProvider` 协议
- `MockSearchProvider` 本地 mock 检索
- `GoogleCustomSearchProvider`，接 Google Programmable Search / Custom Search JSON API
- `BaiduHtmlSearchProvider`，实验性直连百度结果页解析
- `BaiduJsonSearchProvider`，给百度官方或商业 JSON SERP endpoint 预留
- `build_query_plan`
- `collect_candidates`
- `decide_source`
- `build_evidence_ledger`
- CLI 命令：`retrieve`

mock provider 不访问网络，只用于验证信息过滤和证据账本结构。

真实搜索接入后，搜索仍然不是为了制造“中立资料库”，而是为了让每个数字人的信息偏食显形。报告页会保留：它搜了什么关键词；它信任、忽略、拒绝了哪些来源；它为什么采信或排斥；哪些证据类型被集体忽略；这些信息阴影如何进入元模型拼图。

## 区域路由

支持以下 provider：

- `mock`：本地确定性检索，不访问网络。
- `model-prior` / `deepseek-prior`：只生成数字人搜索前的模型先验层；它不是外部事实，用于显影默认记忆、历史类比、价值排序和注意力偏向。
- `google`：使用 Google Custom Search JSON API，需要 `GOOGLE_API_KEY` 和 `GOOGLE_CSE_ID`。
- `baidu-html`：实验性解析百度结果页，不需要 key，但不适合生产稳定依赖。
- `baidu-json`：请求 `BAIDU_SEARCH_ENDPOINT`，用于接入百度官方或商业 SERP JSON API。
- `auto-cn`：中国场景，优先 `baidu-json`，否则 `baidu-html`。
- `auto-global`：海外场景，配置 Google key 时使用 `google`，否则回退 `mock`。
- `hybrid-mock`：模型先验层 + 本地模拟搜索，适合开发验证。
- `hybrid-cn`：模型先验层 + 中国搜索路由。
- `hybrid-global` / `hybrid`：模型先验层 + 海外搜索路由。
- `hybrid-google` / `hybrid-baidu-json` / `hybrid-baidu-html`：模型先验层 + 指定外部搜索源。

环境变量：

```powershell
$env:GOOGLE_API_KEY="your-google-key"
$env:GOOGLE_CSE_ID="your-google-cse-id"
$env:BAIDU_SEARCH_ENDPOINT="your-baidu-json-serp-endpoint"
$env:BAIDU_SEARCH_API_KEY="your-baidu-or-serp-key"
$env:BAIDU_SEARCH_QUERY_PARAM="q"
$env:BAIDU_SEARCH_RESULTS_PATH="response.docs"
$env:BAIDU_SEARCH_TITLE_FIELD="title,name,headline"
$env:BAIDU_SEARCH_URL_FIELD="url,link,target"
$env:BAIDU_SEARCH_SNIPPET_FIELD="snippet,summary,description,abstract"
```

设计上，模型先验和真实搜索都会进入同一本证据账本，但必须保留 `retrieval_layer` 与 `verification_status`。真实搜索 provider 只负责返回候选来源；是否采信、排斥或忽略，仍然由每个数字人的信息过滤器决定。

系统还会在本地登记规范化网址、发布者和原始稿件指纹。不同网址、不同媒体转载同一稿件，仍只算一个信息来源。外部标题和摘要一律作为不可信数据处理；其中出现的任何指令都不能修改数字人的身份、任务、系统规则或输出格式。
