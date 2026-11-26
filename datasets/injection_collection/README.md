# 注入条件轨迹的原始记录

`pair_manifest.jsonl` 冻结正常组的 1,761 个配对位置。`coverage.json` 是归档入口接入前的来源核对。已归档的原始日志保存在 `raw_traces/`，文件名用对应的正常运行 ID。

从仓库根目录传入现成的 JSONL 日志，或传入包含 `trace.jsonl` 的目录。

```bash
python3 -B -m mas.import_injection_traces /path/to/saved/traces
```

入口核对原始题目、MAS、模型运行参数、声明的注入条件和消息引用。日志附带数据集版本时一并校验。运行配置中的注入条件只表示该运行的登记条件。实际暴露仍需从记录中的输入事件确认。归档保留原始字节及已有事件。记录器不生成句级标注。命令输出每份日志的归档路径或拒绝原因。
