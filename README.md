# python-agent -- an example agent for participant-agent-protocol-v4

[中文说明见 README.zh.md](README.zh.md)

A small, Codex-style reference implementation for the GOSIM survey26 hackathon's
telescope-survey challenge. It speaks `participant-agent-protocol-v4` (one JSON object
per line on stdin/stdout) and is organized as a real multi-module project instead of a
single script, so you can see where each concern lives and lift whichever part you need.

It runs a real anchor-search planner (rank candidates, try each fibre as the pointing
centre, fill the other 15 with the best-value neighbours, size the exposure and pick a
program by expected rate) -- the same algorithm as this repo's companion TypeScript
example, so the two are directly comparable -- plus two LLM-advised planning calls
each night. Needs an API key to run; see Configuration below. Read it, run it, and
build your own strategy from the parts that are useful to you.

## Layout

```
agent.py                 entry point: the stdin/stdout loop, one message type per branch
agent_core/
  protocol.py             transport: read one JSON object per line, write one back
  state.py                 SurveyState: catalogue + everything tracked across decisions
  geometry.py              public sky maths: sidereal time, alt/az, the 16-fibre grid
  scoring.py               factor/score estimates from PUBLIC scoring config only
  planner.py               decision logic: wait / observe / report / finish
  llm_client.py            an OpenAI-compatible chat client, defaults to Kimi Coding Plan
  memory.py                an optional, best-effort JSONL decision trace (off by default)
  validation.py            protocol-legal action checking + a deterministic fallback
observer.project.json      the platform's project manifest (image, run command, env)
requirements.txt           none needed -- standard library only
.env.example               copy to .env and set an API key before running
pack_agent.py              zip this folder into a project ZIP for submission
```

## Why these modules

- **protocol.py** -- isolates the wire format (JSON-per-line, stdout-only-for-protocol,
  stderr-only-for-logs) so nothing else in the project has to think about it.
- **state.py** -- the single source of truth for what the agent currently knows: the
  target catalogue from `initialize` (as parallel arrays, one slot per target), a
  declination-band spatial index for fast "what's near this point" queries, per-target
  `factor`/`misses`/`attempts`, a learned sky `scale` (the median of recent quality
  samples backed out of the agent's own hits -- the hidden instrument/transparency/sky/
  seeing terms are never read from a file), weather notices, and fault-diagnosis
  history. Built once, updated on every `decision_request`.
- **geometry.py** -- the public sidereal-time / alt-az / fibre-grid / lunar-factor
  formulas from the participant guide's Geometry and Scoring sections. Any agent
  needs these; they involve no scenario data.
- **scoring.py** -- the *public* parts of `initialize.payload.scoring` (`q0`,
  `flux_zero_point`, `exposure_zero_point_seconds`, `airmass_exponent`, `lunar_model`,
  `program.{bands,multipliers,mismatch_multiplier}`, `required.*`, `uniformity.*`) turned
  into factor/band/multiplier estimates.
- **planner.py** -- an anchor-search strategy: rank visible not-yet-done targets (a
  required target not yet past the public factor threshold gets a large bonus; targets
  setting soon or with few nights left rank higher); for the best few candidates, try
  every fibre as the pointing centre and fill the rest with the best-value neighbours
  that land on the glass; keep the best pointing; pick the exposure length with the best
  expected score per second and the program most assignments will match. Sleeps through
  the day, closes the shutter on an all-sky rain/storm bulletin, and reports a suspected
  instrument fault only after a sustained, unexplained quality drop confirmed over
  several nights.
- **llm_client.py** -- an OpenAI-compatible client (plain `urllib`, no SDK) defaulting
  to the Kimi Coding Plan endpoint, configurable to any other OpenAI-compatible
  `/chat/completions` endpoint by environment variable.
- **memory.py** -- an optional, best-effort JSONL decision trace (night advice, the
  finish summary), off by default. The state/planner's own working memory (learned
  scale, per-target progress, night advice, report cooldowns) lives on
  `SurveyState`/`Planner` themselves, since that is genuinely part of what they track.
- **validation.py** -- the last line of defense before anything reaches stdout: checks
  every field the protocol actually validates (ranges, duplicate fibres/targets, unknown
  fields, the consecutive-report limit) and provides `fallback_action()`, a response
  that is always legal no matter what state the agent is in.

## Where the LLM is used

Once at the start of every night (`Planner._night_advice`), two questions go to the
model, each through `LLMClient.ask_json`, each answered as
`{avoid_directions, duration_scale}`:

1. **Forecast call** -- reads tonight's forecast notices plus the current bulletin, and
   asks for directions to avoid and an exposure-length scale.
2. **Bulletin + hit-rate call** -- reads tonight's live bulletin as plain text, plus the
   agent's own hit rate so far this run, and asks the same question from that angle.

The two answers are merged: directions to avoid are unioned, and the duration scale is
averaged. If a call keeps failing after its retries, that question's answer is left out
of the merge for the night (the other one still counts if it succeeded); the next
night's calls are unaffected. Target selection, fibre filling, exposure sizing, and
instrument-fault reporting are otherwise fully deterministic -- calling a model on every
one of the ~1,000-5,000 decisions in a run would blow through the wall clock; the
participant guide's own advice is "do not call a model on every decision." A third,
occasional call asks the model to confirm a suspected instrument fault before reporting
one (at most twice per run; the rule-based evidence check that triggers it runs far less
often than once per night).

### Call behaviour

`LLMClient` enforces, regardless of what the model does:

- a per-call timeout (default 12 s) that also shrinks as the wall clock runs low (never
  cuts into the last 60 s of the survey's own budget);
- a total LLM time budget per run (default 300 s, well under the 900 s wall clock);
- a call cap per run (default 100);
- up to 3 attempts for one question before moving on without an answer for it this time;
  the next scheduled call still goes ahead as normal;
- every exception (network, timeout, bad JSON, missing fields) is caught -- a failed
  attempt never raises past `ask_json`, and `ask_json` returns `None` when all attempts
  for that question are exhausted;
- the HTTP `User-Agent` header is never set or overridden.

## Configuration (.env)

Copy `.env.example` to `.env` and set an API key before running locally:

```
OPENAI_API_KEY=sk-...
```

That alone is enough: the client defaults to the
[Kimi Coding Plan](https://www.kimi.com/code/docs/en/) endpoint
`https://api.kimi.com/coding/v1` with model `k3`. Accounts outside mainland China
should instead use `https://api.kimi.ai/coding/v1`:

```
OPENAI_BASE_URL=https://api.kimi.ai/coding/v1
OPENAI_API_KEY=sk-...
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` override the defaults, so any other
OpenAI-compatible `/chat/completions` endpoint works too (OpenAI itself, a local proxy,
etc.) -- just point them there. `KIMI_API_KEY` is accepted as an alternate name for the
key if you'd rather set that. On the platform, `OPENAI_BASE_URL` / `OPENAI_API_KEY` are
injected automatically (the platform's own model proxy and a temporary credential); you
never need to configure them for a submission, and `.env` is never packed into the ZIP.

Without an API key (`OPENAI_API_KEY` or `KIMI_API_KEY`) configured, the process checks
at startup, before reading anything from stdin, and exits with a message on stderr and
a non-zero exit code.

## Running it locally

This project is agent-side only -- it does not ship the simulator/scorer. Test it with
any tool that speaks the protocol on stdin/stdout (a local runner, or the platform
itself), pointing it at this folder's `agent.py`. See `docs/` for the full protocol and
scoring reference.

`agent.py` reads `initialize` / `decision_request` / `finish` on stdin and writes
`decision_response` on stdout; everything else (logs) goes to stderr, matching the
protocol exactly.

## Packaging / submitting

```bash
python3 pack_agent.py --out ../python-agent.zip
```

This zips the project with `observer.project.json` at the root
(`"protocol": "jsonl-v4"`, `run: ["python3", "-u", "agent.py"]`), skipping `.env`,
`__pycache__`, and `run_output/`. Upload the ZIP as a complete project, or push this
folder to a GitHub repository. No third-party packages are required; if you add one,
list it in `requirements.txt` **and** add a matching `build` step to
`observer.project.json` -- dependencies are not installed automatically otherwise.

## Safety properties

- Never reads any file: all state comes from stdin.
- Every decision is wrapped in `try`/`except` in `agent.py`; a planner bug produces a
  safe `wait`, never a crash or an `agent_error`.
- `validate_action()` strips anything the protocol would reject before it reaches
  stdout.
- A planning call that keeps failing just leaves that question's answer out of the
  merge for the night; the rest of the decision loop (targeting, exposure sizing,
  validation) runs the same either way.

## License / Citation

Task cards, simulated data, evaluation code and this example project are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution, non-commercial);
please cite the GOSIM 2026 Agentic Observer Challenge. Your own agent code is not restricted
by this. See `LICENSE.md` for details.
