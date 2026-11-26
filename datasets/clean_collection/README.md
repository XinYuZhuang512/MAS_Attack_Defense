# 正常条件轨迹的任务清单

三个系统共用这三份 JSONL，每个系统每题运行一次。

| 清单 | 独立任务数 | 选择规则 |
|---|---:|---|
| `pear.jsonl` | 123 | 固定提交的四域注册任务。银行、Slack、旅行各 30 条，办公 33 条。旅行注册表后续还有编号 30–36 的七个任务，该批次按 123 条的范围冻结。 |
| `humaneval.jsonl` | 164 | OpenAI HumanEval 原版完整题集。测试代码单独保存，模型输入只包含题干。 |
| `mmlu.jsonl` | 300 | CAIS MMLU 的 `all/test`。在人文和社会科学学科内去重并近似均匀抽样，随机种子为 20261125。 |

`sources.json` 记录固定版本、文件哈希和抽样配置。正常条件下的任务失败和工具参数错误仍保留在轨迹中。采集时只有 PEAR 使用任务模拟工具。HumanEval 的功能评估单独运行，采集阶段保留原始答案。

完成的轨迹可在本目录的 `trajectories/<system>/<dataset>/<task_id>/` 查阅。每个任务目录包含 `trace.jsonl`、`annotation_inputs.jsonl` 和 `result.json`。轨迹属于本地采集数据，体量较大，已在这一层配置 Git 忽略。
