# python-agent —— participant-agent-protocol-v4 的示例智能体

[English README see README.md](README.md)

这是 GOSIM survey26 黑客松望远镜巡天赛题的一个小型、Codex 风格参考实现。它说
`participant-agent-protocol-v4`(stdin/stdout 上每行一个 JSON 对象),并按真实的多模块
项目来组织,而不是单文件脚本,方便你看清每一部分职责所在,按需取用。

它跑的是一个真正的锚点搜索(anchor-search)规划器:给候选目标排序、逐一尝试把每根
光纤当作指向中心、用能落在玻璃上的最有价值邻居填满剩下 15 根光纤、按期望速率选曝光
时长与项目——与本仓库同伴 TypeScript 示例完全同一套算法,两者可以直接对比——再加上
每晚两次 LLM 建议调用。运行需要一个 API key,见下面的「配置」。读懂它、跑起来,再从
中挑选对你有用的部分搭自己的策略。

## 目录结构

```
agent.py                 入口:stdin/stdout 主循环,按消息类型分支处理
agent_core/
  protocol.py             传输层:逐行读一个 JSON 对象,逐行写一个 JSON 对象
  state.py                 SurveyState:星表 + 跨决策需要跟踪的一切状态
  geometry.py              公开天球几何公式:恒星时、地平坐标、16 根光纤网格
  scoring.py               只用公开评分配置估算 factor/score
  planner.py               决策逻辑:wait / observe / report / finish
  llm_client.py             OpenAI 兼容 chat 客户端,默认对接 Kimi Coding Plan
  memory.py                 可选的、尽力而为的 JSONL 决策轨迹日志(默认关闭)
  validation.py             协议合法性校验 + 确定性兜底动作
observer.project.json      平台项目清单(镶像、运行命令、环境变量)
requirements.txt           无需任何第三方包——只用标准库
.env.example               复制为 .env,运行前先配好 API key
pack_agent.py              把本目录打包成可提交的 ZIP
```

## 为什么这样拆模块

- **protocol.py** —— 把线上协议格式(逐行 JSON、stdout 只给协议用、日志只走 stderr)
  隔离出来,其余代码完全不用操心这件事。
- **state.py** —— 智能体当前所知一切的唯一来源:`initialize` 给的星表(按并行数组
  存储,每个目标一个下标)、按赤纬分带的空间索引(快速查"这附近有什么")、每个
  目标的 `factor`/`misses`/`attempts`、学到的天空 `scale`(从自己命中反推出的质量
  样本的中位数——隐藏的仪器/透明度/天空/视宁度项绝不从任何文件读取)、天气通知、
  故障诊断历史。只在 `initialize` 建一次,每次 `decision_request` 更新一次。
- **geometry.py** —— 来自参赛者指南 Geometry 与 Scoring 小节的公开公式:恒星时、
  地平坐标转换、光纤网格判定、月光因子。任何智能体都需要这些,且与具体场景
  数据无关。
- **scoring.py** —— `initialize.payload.scoring` 中**公开**的部分(`q0`、
  `flux_zero_point`、`exposure_zero_point_seconds`、`airmass_exponent`、
  `lunar_model`、`program.{bands,multipliers,mismatch_multiplier}`、`required.*`、
  `uniformity.*`)转换成 factor/波段/倍数估算。
- **planner.py** —— 锚点搜索策略:给可见、尚未完成的目标排序(必做且尚未过公开
  factor 门槛的目标拿到一个大加成;快要落下或剩余夜数不多的目标排名更高);对排
  名靠前的几个候选,逐一尝试把每根光纤当作指向中心,用能落在玻璃上的最有价值邻居
  填满其余光纤,保留最好的指向;选能带来最佳期望得分/秒的曝光时长,以及最多目标
  会匹配上的项目。白天整段 `wait` 过去;公告里出现全天区 rain/storm 就关闸;只有在
  质量出现持续、无法解释的下降并在多个夜晚得到确认后,才上报可疑的仪器故障。
- **llm_client.py** —— OpenAI 兼容客户端(纯 `urllib`,不依赖任何 SDK),默认对接
  Kimi Coding Plan 接口,改环境变量即可换成任何其它 OpenAI 兼容 `/chat/completions`
  接口。
- **memory.py** —— 可选的、尽力而为的 JSONL 决策轨迹日志(今晚建议、结束汇总),
  默认关闭。state/planner 真正跨决策携带的记忆(学到的 scale、每个目标的进度、
  今晚建议、report 冷却)放在 `SurveyState`/`Planner` 自己身上,因为那本就是它们
  各自该跟踪的状态。
- **validation.py** —— 发往 stdout 前的最后一道防线:检查协议真正会校验的每一项
  (数值范围、光纤/目标重复使用、未知字段、连续 report 次数上限),并提供
  `fallback_action()`——无论智能体处于什么状态,这个回复永远合法。

## LLM 用在哪里

每晚开始时(`Planner._night_advice`)都会问模型两个问题,都经过
`LLMClient.ask_json`,都回答成 `{avoid_directions, duration_scale}`:

1. **预报调用**——读取今晚的预报通知加当前公告,请模型给出要避让的方向和曝光
   时长系数。
2. **公告+命中率调用**——把今晚的公告转成一段文字,配上智能体本局至今的命中率,
   从这个角度再问一遍同样的问题。

两份回答会合并:要避让的方向取并集,曝光时长系数取平均。如果某次调用在重试后仍然
失败,这个问题当晚就不计入合并(如果另一次成功了,仍按那次的结果算);下一晚的调用
不受影响。目标选择、光纤填充、曝光时长、仪器故障上报的其余逻辑保持完全确定性——
一局里有约 1,000–5,000 次决策,每次都调模型会直接把钟表时间耗尽,参赛者指南自己也
写明"不要每次决策都调模型"。还有第三个、不常见的调用,用来在上报可疑仪器故障前
请模型确认一下(每局最多两次;触发它的规则性证据检查本身就比每晚一次少得多)。

### 调用行为

无论模型怎么回,`LLMClient` 都会:

- 单次调用超时(默认 12 秒),且随钟表时间紧张而收紧(绝不占用生存钟表最后 60
  秒的预算);
- 整局 LLM 总耗时预算(默认 300 秒,远低于 900 秒钟表上限);
- 每局调用次数上限(默认 100);
- 一个问题最多尝试 3 次,这次没问出答案就先不用,下一次该问的时候照常再问;
- 捕获任何异常(网络、超时、JSON 解析失败、字段缺失)——失败的尝试不会从
  `ask_json` 内部抛出来,所有尝试都用完时 `ask_json` 返回 `None`;
- 绝不设置或改写 HTTP 的 `User-Agent` 头。

## 配置(.env)

把 `.env.example` 复制为 `.env`,运行前先配好一个 API key:

```
OPENAI_API_KEY=sk-...
```

这样就够了:客户端默认对接
[Kimi Coding Plan](https://www.kimi.com/code/docs/en/) 接口
`https://api.kimi.com/coding/v1`,模型用 `k3`。非中国大陆账号应改用
`https://api.kimi.ai/coding/v1`:

```
OPENAI_BASE_URL=https://api.kimi.ai/coding/v1
OPENAI_API_KEY=sk-...
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` 会覆盖上面的默认值,所以任何其它 OpenAI 兼容的
`/chat/completions` 接口(OpenAI 本身、本地代理等)改两个变量就能用。key 也可以用
`KIMI_API_KEY` 这个别名来设。在平台上,`OPENAI_BASE_URL` / `OPENAI_API_KEY` 会被
自动注入(平台自己的模型代理和一次性临时凭证),提交时完全不需要自己配置,`.env`
也永远不会被打进提交 ZIP。

没配 API key(`OPENAI_API_KEY` 或 `KIMI_API_KEY`)时,进程会在启动时、读任何 stdin
之前就检查这一点,然后把错误信息打到 stderr,以非零退出码退出。

## 本地运行

本项目只是智能体一侧——不含模拟器/评分器。请用任何支持该协议(stdin/stdout)的
本地运行器,或平台本身,把 `--agent` 指向本目录的 `agent.py`。完整协议与评分公式
见 `docs/`。

`agent.py` 在 stdin 上读取 `initialize` / `decision_request` / `finish`,在 stdout 上
写 `decision_response`;其余内容(日志)全部走 stderr,与协议要求完全一致。

## 打包 / 提交

```bash
python3 pack_agent.py --out ../python-agent.zip
```

会把本项目打包,`observer.project.json` 置于 ZIP 根目录
(`"protocol": "jsonl-v4"`、`run: ["python3", "-u", "agent.py"]`),并跳过 `.env`、
`__pycache__`、`run_output/`。把 ZIP 作为完整项目上传,或把本目录推到 GitHub 仓库。
不需要任何第三方包;如果你加了依赖,请同时写进 `requirements.txt` **并**在
`observer.project.json` 里加一个匹配的 `build` 步骤——否则依赖不会被自动安装。

## 安全属性

- 从不读取任何文件:所有状态都来自 stdin。
- `agent.py` 里每个分支都包在 `try`/`except` 中;planner 出 bug 只会得到一个安全的
  `wait`,绝不会崩溃或变成 `agent_error`。
- `validate_action()` 会在写入 stdout 之前,剥掉一切协议会拒绝的内容。
- 一次规划调用持续失败,只会让那个问题的答案这一晚不计入合并;其余决策流程
  (选目标、算曝光、校验)照常运行。

## 许可 / 引用

任务卡、模拟数据、评测代码与本示例项目按 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
提供（署名、非商业）；使用请引用 GOSIM 2026 Agentic Observer Challenge。选手自己编写的代码不受此限制。
详见 `LICENSE.md`。
