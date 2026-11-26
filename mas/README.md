# MAS 源码与记录

`upstream/` 保存 PyPI 官方源码发行包，版本、下载地址和 SHA-256 见该目录的 `sources.json`。AutoGen 分为 core 与 agentchat 两个包。MetaGPT 用于后续跨系统评估。

`workflows.py` 提供 AutoGen 的一分二再汇合配置，以及 CAMEL 的双角色配置。调用者提供已配置的 agent 或模型。框架依赖可分别安装在独立环境中。

`trace.py` 提供单进程调度器的 JSONL 记录接口。调度器创建消息后，按实际收件人逐个调用 deliver。每次模型调用前用 start_call 保存有序的实际输入及触发消息 ID，生成时用 reasoning 保存框架暴露的文本，结束时用 finish_call 引用输出消息。流式摘要标为 summary，无可见推理标为 unavailable。两个分支汇合时保留两个触发消息 ID。时间戳指记录器收到事件的本机时间。运行适配器需在实际消息路由和模型调用边界接入这些接口。

## 跑通数据采集

从仓库根目录运行，输出目录需尚未存在。

```bash
python3 -B -m mas.experiment smoke --out-dir tmp/mas_offline_trial
python3 -B -m mas.experiment deepseek --out-dir tmp/mas_flash_trial
```

`smoke` 使用固定文本检查消息路由和记录。 `deepseek` 读取环境变量 `DEEPSEEK_API_KEY`，以 `deepseek-flash` 的思考模式运行同一四角色分支。两条命令均可用 `--task-id` 选择 `datasets/pear/pilot_tasks.jsonl` 中的任务。默认使用 `pear/workspace/user_task_1`。如果当前 Python 缺少默认 CA 文件，可以使用本机受信任的 CA，例如 macOS 上设置 `SSL_CERT_FILE=/etc/ssl/cert.pem`。

这个入口使用本仓库的调度器。Flash 试跑只发送公开任务、角色说明和已交付的上游输出，未加载 PEAR 环境或注册任务工具。试跑用于验证数据链路，PEAR 完成率需要在实际任务环境中单独测量。

每次生成两份文件。

- `trace.jsonl` 记录原始 Q、调用、可见推理片段、答案片段及消息创建与交付事件。跨分支输入会校验其交付记录。
- `annotation_inputs.jsonl` 从实际输入与当前调用的可见推理生成逐句前缀，保存原文区间和句尾字符所在的流式事件。每一行可单独提交给标注器，现阶段仅含标注输入。

已有轨迹可以离线重放。默认按英文标点初切，有 Stanza 资源时可加 `--splitter stanza --stanza-dir /path/to/resources`。初切边界需要在句级金标复核时校正。句尾字符的接收时间用于回放，实时预警提前量需要另行测量判定证据到达的时间。

```bash
python3 -B -m mas.experiment prepare tmp/mas_flash_trial/trace.jsonl --output tmp/mas_flash_trial/replay_inputs.jsonl
```

## AutoGen 和 PEAR 正常轨迹

先在实验环境安装依赖，从仓库根目录运行下列命令。

```bash
python3 -m pip install -r mas/requirements-autogen.txt
python3 -B -m mas.experiment autogen-pear --out-dir tmp/autogen_pear_clean
```

入口使用 AutoGen 0.7.5 的 GraphFlow 调度四个 agent。规划员分派给日历查询和需求核对两个角色，汇总员等两条分支都完成后作答。MessageFilterAgent 按来源限制各角色看到的消息。日志里的输入是过滤后实际发送给 DeepSeek 的输入。

任务固定为 PEAR Workspace 的 `user_task_3`。日历分支可调用 `search_calendar_events` 查询官方 Workspace YAML。查询适配器复现上游的标题或描述匹配和可选日期过滤，工具结果提供任务所需的事件字段。最终检查答案是否包含官方期望地点，并验证环境文件哈希。该入口只接入这个只读任务的环境和工具，PEAR 的完整任务执行器没有接入。[PEAR 官方工具](https://github.com/MingxuanZhangPurdue/multiagent-vulnerable/blob/cf69c68ccb1a532d3012f58f18ef7a05554d6fe3/src/mav/Tasks/workspace/tools.py)

DeepSeek Flash 显式开启思考模式，`reasoning_effort=high`，每次调用的 `max_tokens=8192`。工具调用前后的请求分别记录。若 API 在某次调用没有返回推理片段，记录为 `unavailable`。轨迹和待标注前缀可用上面的 `prepare` 命令复核。


## 正常轨迹批量采集

`datasets/clean_collection` 中为三个系统冻结同一批 PEAR、HumanEval 和 MMLU 任务。批量入口是 `python3 -B -m mas.collect_clean SYSTEM --datasets pear humaneval mmlu --output OUTPUT`。三套工作流分别使用 AutoGen GraphFlow、CAMEL RolePlaying 和 MetaGPT Team，统一调用 DeepSeek Flash 的 `high` 思考强度。当前每次模型调用的输出上限为 8192 tokens。

PEAR 使用独立的本地模拟状态。`MAS_PEAR_SOURCE` 指向固定 PEAR 提交中提供 `mav/Tasks` 的 `src` 目录，Python 环境需有 PyYAML、Pydantic、email-validator 和 deepdiff。外部框架依赖分别安装在独立虚拟环境。CAMEL 0.2.90 需要 MCP 1.x。MetaGPT 的兼容依赖记录与运行日志一同存档。

每个任务结束后先校验消息引用并导出逐句前缀，然后再发布结果目录。重启时跳过已有完整结果。API 中断和截断会保留在 attempts 中，每题最多尝试三次。进度日志每 30 分钟记录一次。模型答错、正常工具失败和没有返回可见推理的调用均保留在原始轨迹中。

续跑会自动跳过已有结果。需要只补某道题时，可重复传入 `--task-id`。对反复遇到输出截断的任务，可用 `MAS_FLASH_MAX_TOKENS` 提高该次运行的输出上限。轨迹的 `experiment_configured` 记录实际值。
