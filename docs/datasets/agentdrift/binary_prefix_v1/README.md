# AgentDrift 二分类前缀实验

[数据文件所在目录](../../../../datasets/agentdrift/binary_prefix_v1)

frames.jsonl 和 replication.jsonl 分别来自官方 val 和 test。两批各 180 条，按五个领域和六种轨迹类型每格固定抽取 6 条，种子分别为 842719 和 842720。排除了旧实验固定样本的轨迹 ID 和上一轮盲评的全部 12 个任务原文。任务文字仍可与更早的实验重复。本次训练规则仅使用先前固定的官方 train 子集，并逐折排除测试领域、任务原文、完整 world 或主用户邮箱相同的训练轨迹。

q 是用户任务的动作列表，x 是每个步骤的动作列表。动作由原形和 role/text 参数构成。y 是保留的原始数据集标签，仅供评估与训练用。当前步 y 为 hijacked 时二分类标签为 1，其余为 0。单步为 0 可以位于一条包含劫持的轨迹之内。task 是用于核查重叠的 SHA-256 摘要。

原始语料由 Asif Pinjari 和 Mithun Paul Saint-Germain 发布，许可为 CC BY 4.0。派生改动包括分层抽样、对原始 task 与 thought 应用本项目句法提取器和保留评估字段。生成脚本与逐步预测位于本项目 tmp/binary_prefix_v1。[原始数据](https://github.com/Asif-0209/AgentDrift) · [许可](../LICENSE)
