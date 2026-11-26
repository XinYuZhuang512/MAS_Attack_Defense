# 英文意图提取

终端和 Python 的默认入口使用 spaCy `en_core_web_trf` 3.8.0。一次加载模型可以连续处理多个用户请求或执行步骤。

```sh
/Users/zhuangxinyu/miniconda3/envs/pai-gen/bin/python -B test/extract_intent.py
```

```python
from intent_extraction import DefaultIntentExtractor

extractor = DefaultIntentExtractor()
result = extractor.extract("Assign the issue to the QA team.")
```

输出按句分组。每个动作保留原词、词元、参数、对象内部的修饰关系和语法作用域。以介词引出的收件方保留原介词，例如 `to the QA team`。token 索引从每句的零号词开始。程序可用 `extract_many` 顺序处理多条输入。正式接口返回 trf 自身的输出，空句返回空的 frames。使用旧的 Stanza 单树对照时指定 `--parser single`；`IntentExtractor`、`chain_frames` 为早期实验的兼容接口。

本机 pai-gen 环境已安装模型。新环境可安装 `spacy>=3.8,<3.9` 后运行 `python -m spacy download en_core_web_trf`。本次验证的配套版本为 spacy 3.8.14、spacy-curated-transformers 0.3.1、curated-transformers 0.1.1、curated-tokenizers 0.0.10。

## 既有多树实验

显式使用 `--parser stanza-beam`、`--parser pcfg`、`--k-best` 或 `--trees-json` 可复现旧实验。现场 Stanza 解析复用 KAL 的本地解码器。可用 `KAL2INU_HOME` 指定 kal2inu 目录，模型位置由该解码器的 `KAL2INU_STANZA_PILOT` 设置。独立的 PCFG 后端仅需要 Python 3.10+、JDK 11+ 和含英文模型及 JSON 库的 CoreNLP 安装。用 `CORENLP_HOME` 和 `JAVA` 指定资源与可执行文件。两种入口均离线运行。

```sh
python3 test/extract_intent.py --parser pcfg --k-best 16 'The signed invoices from April were archived by Morgan.'
python3 test/extract_intent.py --trees-json /path/to/exported_trees.json --full
```

批量处理已分句的输入时，复用 `StanzaBeamExtractor(k=16)`，调用 `extract_many(sentences)`。PCFG 后端的 `PCFGIntentExtractor.extract_many(texts)` 会在同一个 JVM 内依次分析输入，并按输入、句子两层返回。原有实验的 `IntentExtractor` 接口和 `--legacy` 仍可用来复现旧结果。

## 多树输出与统计口径

每条候选树给出成分跨度、词性、基本依存边以及原词跨度。每个谓词带原词、词元、参数和语法作用域。参数保留中心词、完整依存短语及其内部成分短语。描述递归挂在被修饰的词下面，例如 `record → nmod:poss → Emily's`。介词短语保留依存角色及原介词。只有同时出现并列依存关系、平行 VP 和自己的宾语时，算法才可在裸动词并列位置恢复被标成名词的谓词。规则依据结构和词性；动词和对象名称都取自输入。

给定同一次搜索返回的树分数 s，树的相对权重等于 exp(s − max(s)) 再归一化。Stanza 束搜索使用不同成分树各自最佳推导的分数，PCFG 使用返回的语法推导分数。两种后端各自在其候选集内归一化。动作和挂接路径按 token 位置累计支持权重。默认展示支持度达到 0.5 的动作。对于同一动作，在真实候选树中选出与所有树加权比较后，缺失和多余挂接总数最少的那个读法。同一谓词下的各字段可以共存。不同谓词可以选中不同树。

`--full` 展示入选动作、低支持度动作、替代挂接路径、所选树序号和局部作用域。`support` 是某动作或挂接路径在当前候选集中的权重份额。挂接的数值对应中心词之间的关系和介词位置；完整短语边界与作用域展示的是所选树的一种读法。K=1 时有效树上所有相关支持度自然为 1。

语法层保留被评价、转述、假设、否定以及描述从句中的谓词，并通过 `scope` 保存上级谓词、否定等原文线索。判断执行者是否采纳、跨句代词解析和攻击分类属于下游任务。Stanza 单树模式只输出 `sentence` 和 `frames`，每个动作保留完整参数与作用域。
