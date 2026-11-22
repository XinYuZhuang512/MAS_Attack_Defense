# AgentDrift 意图投影样本

[数据文件所在目录](../../../../datasets/agentdrift/intent_coherence_pilot)

frames.jsonl 含 1,110 条轨迹。在官方 train 和 test 中分别按领域和轨迹类型固定抽取 750 和 360 条。每行的 q 为用户任务，x 是按执行顺序排列的各步骤，每个动作的结构为 [action, [[role, text], ...]]。当前提取器可能为某个任务或步骤返回空列表。y 存放原数据的逐步标签，id、pool、domain、category 用于划分或审计，task、world、user 是去重用的 SHA-256 摘要。这些评估字段均不输入比较器。

源数据由 Asif Pinjari 和 Mithun Paul Saint-Germain 发布，遵循 CC BY 4.0。此处的改动包括抽样、用项目句法提取器投影任务和 thought、对三项身份字段散列，保留原始逐步标签。[上游项目](https://github.com/Asif-0209/AgentDrift) · [许可](../LICENSE)

提取时的项目提交为 6b76155f04a4b10a63c9c69ae9fb168304c9606d，提取器文件 SHA-256 为 f2d6b4647fd0e27e6c491c196316c086d9404c28e0f964db47850398e6bb560e。此版本 frames.jsonl 的 SHA-256 为 1c6a9bf906729cce07b0d6ca1b36909e90aea7e57310dd886b14b0c3fc370a55。
