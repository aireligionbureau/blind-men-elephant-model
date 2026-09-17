# 标准版真实批量运行

本阶段把模型推进到标准版 24 数字人的真实运行流程：

1. 为问题生成 24 个新数字人。
2. 每个数字人按自己的信息过滤器检索和形成证据账本。
3. 并发调用配置的兼容大模型，让每个数字人只按自己的认知身份证思考。
4. 对失败调用自动重试。
5. 每个数字人输出单独保存，避免中途失败损失已有结果。
6. 汇总 token 和费用。
7. 元模型读取数字人输出与证据账本，生成第一版诊断。
8. 阴影拼图引擎把阴影转成拼图材料。

## 运行命令

PowerShell:

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="your-api-key"
$env:BME_INPUT_PRICE_PER_1M_CNY="your-input-price"
$env:BME_OUTPUT_PRICE_PER_1M_CNY="your-output-price"

python -m bme_model run-standard-live "是否应该加快部署自动驾驶卡车？" `
  --provider mock `
  --model deepseek-v4-pro `
  --max-tokens 4096 `
  --concurrency 24 `
  --retrieval-concurrency 12 `
  --semantic-concurrency 24 `
  --model-concurrency 24 `
  --retries 2 `
  --output-dir runs
```

如果已配置真实搜索：

```powershell
python -m bme_model run-standard-live "是否应该加快部署自动驾驶卡车？" --provider auto-cn
python -m bme_model run-standard-live "是否应该加快部署自动驾驶卡车？" --provider auto-global
```

如果要把“脑中先验 + 带着过滤器搜索 + 选择性采信”合在一起：

```powershell
python -m bme_model run-standard-live "是否应该加快部署自动驾驶卡车？" --provider hybrid-cn
python -m bme_model run-standard-live "是否应该加快部署自动驾驶卡车？" --provider hybrid-global
python -m bme_model run-standard-live "是否应该加快部署自动驾驶卡车？" --provider hybrid-mock
```

如果已经完成一次 24 人批量运行，后续升级元模型诊断、立场分类或阴影拼图算法时，可以直接重建保存结果，不重复消耗 API：

```powershell
python -m bme_model rebuild-run runs\20260701-171851-standard
```

## 输出文件

每次运行会生成一个目录：

```text
runs/YYYYMMDD-HHMMSS-standard/
  retrieval.json
  person_outputs/
    <person_id>.json
  diagnosis.json
  shadow_puzzle.json
  summary.json
  run.json
  report.html
```

`report.html` 是给人看的结果页。它的结构是：

1. 先展示“这群盲人摸出了一个怎样的轮廓”。
2. 再展示多人共同摸到的方向和共同漏看的事实。
3. 最后展开每个数字人的认知身份证、信息过滤器、先验层、搜索层、采信/忽略记录和元模型指出的思维阴影。
3. 最后展开“每个数字人是怎么摸、哪里摸偏了”。

附录会显示每个数字人的身份、信息过滤器、搜索关键词、采信/忽略/拒绝来源、核心假设、推理路径、结论、元模型指出的问题，以及该阴影如何进入拼图。

## 元模型第一版诊断

`diagnosis.json` 包含：

- `disagreement_structure`：身份分歧与输出分歧。
- `blind_spot_registry`：盲区登记。
- `bias_attribution`：偏见归因。
- `collective_blind_spots`：集体盲区。
- `differential_diagnosis`：透镜竞争与误诊审计。
- `readiness_evaluation`：当前结构强度评分。

## 阴影拼图引擎

`shadow_puzzle.json` 包含：

- `interlocking_anchors`：互锁锚点。
- `bias_corrected_fragments`：偏矫正碎片。
- `negative_space_holes`：负空间洞。
- `fracture_boundaries`：推理断裂边界。
- `next_probe`：审计用的未决缺口记录；不触发自动第二轮，也不作为默认结果页环节。

## 费用统计

费用统计依赖 API 返回的 token usage，以及你配置的单价：

- `BME_INPUT_PRICE_PER_1M_CNY`
- `BME_OUTPUT_PRICE_PER_1M_CNY`

旧的 `DEEPSEEK_INPUT_PRICE_PER_1M_CNY` 和 `DEEPSEEK_OUTPUT_PRICE_PER_1M_CNY` 仍作为兼容回退。

如果未配置单价，系统仍会统计 token，但 `estimated_cost_cny` 会为空。
