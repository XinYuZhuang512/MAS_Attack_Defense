# 实验条件与观察数据

`conditions.json` 给出先导实验的正常、工具结果、角色配置和委派消息条件及文献出处。`conditions.py` 生成配对运行的研究元数据，用于保存正式基准病例 ID 和受影响对象，独立于 agent 可见的输入。

`agentdrift_step_index.jsonl` 引用本仓库 AgentDrift 原始轨迹中的注入步、首个 hijacked 步和失败注入步，步号从 1 开始。首个 hijacked 步为 null 表示该轨迹无此标签。旧数据为步骤级标注，MAS 的首次可见采纳句需要另行采集和标注。

在仓库根目录运行 `python3 attacks/step_labels.py datasets/agentdrift/data` 可将同一索引输出到标准输出。

`pear_privacy_cases.jsonl` 是四个领域共 40 个 PEAR 隐私病例的注册类索引。每条给出领域、上游文件、类名、Git 提交和文件 blob，可作为 run_condition 的病例来源。索引本身不包含病例执行、自动注入或成功判定逻辑。
