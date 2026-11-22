# 分支上下文实验材料

[数据文件所在目录](../../../../datasets/agentdrift/branch_context_v1)

cases.json 有 10 组成对的开发例和 13 条边界对照；transfer_cases.json 另外提供 6 组成对的新表达。它们是本次研究手工撰写的英文诊断样本。每一对的用户任务和最终句完全相同，期望输出依次为 0 和 1。history、route、sister、local 表示运行时发生的先前思考、委派消息或分支活动，expected 和 expected_reason 仅供评估。

test_projection.jsonl 来自 AgentDrift 官方 test 的全部轨迹。q 是根任务的增强动作投影，x 是按执行顺序排列的各步投影。没有任何交付动作同时提取到对象与接收方的根任务，x 记为 null，因为这类任务不可能建立本实验要检查的接收方约束。评估仍保留其每一步和原始标签。qualified 只用于防止把带例外或条件的根任务误写成无条件限制，exclusive 标记本原型识别到的接收方范围。其余角色和文本来自项目已有 Stanza 提取器，实验适配层补充了局部 only 的作用域、share with 的接收方分析等。

原始数据来自 Asif Pinjari、Mithun Paul Saint-Germain 的 AgentDrift（CC BY 4.0）。本次派生改动包括意图投影、逐步编号、保留参考标签，以及按上述适用条件省略部分步骤的文本特征。[上游数据](https://github.com/Asif-0209/AgentDrift) · [许可](../LICENSE)
