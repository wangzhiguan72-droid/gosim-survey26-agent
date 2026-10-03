# AGENTS.md -- python-agent

Guide for AI coding assistants working in this project. Humans: see README.md /
README.zh.md.

## What this is

A complete, standard-library-only Python agent for the GOSIM survey26 hackathon's
telescope-survey challenge. It speaks `participant-agent-protocol-v4` (one JSON
object per line on stdin/stdout), runs a deterministic anchor-search planner, and
calls an LLM twice per night for forecast/bulletin advice plus occasionally to
confirm a suspected instrument fault. Read `agent.py` first -- it is the thin
stdin/stdout loop that dispatches to everything else.

## Module map

```
agent.py                 entry point: stdin/stdout loop, one branch per message type
agent_core/
  protocol.py             transport: read/write one JSON object per line
  state.py                 SurveyState: catalogue + everything tracked across decisions
  geometry.py              public sky maths (sidereal time, alt/az, fibre grid)
  scoring.py               factor/score estimates from PUBLIC scoring config only
  planner.py               decision logic: wait / observe / report / finish
  llm_client.py            OpenAI-compatible chat client, defaults to Kimi Coding Plan
  memory.py                optional, best-effort JSONL decision trace (off by default)
  validation.py            protocol-legal action checking + a deterministic fallback
observer.project.json      platform project manifest (image, run command, env)
requirements.txt           none needed -- standard library only
pack_agent.py              zip this folder into a project ZIP for submission
```

## Changing the strategy

Target ranking, fibre filling and exposure sizing live in `agent_core/planner.py`.
The two LLM-advised calls (forecast avoidance + bulletin check-in) are issued from
the night-advice path in `planner.py` through `agent_core/llm_client.py`; the
instrument-fault confirmation call is a third, rarer call from the same module. You
can change the ranking heuristic, add new signals to `SurveyState`, change what (if
anything) the LLM is asked, or replace the LLM calls entirely with a different
approach -- the only hard requirement enforced by `validation.py` is that whatever
`agent.py` writes to stdout is a protocol-legal response.

## Configuring the LLM (Kimi key / base URL / model)

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY` (or `KIMI_API_KEY`) -- required to run the bundled planner as-is.
- `OPENAI_BASE_URL` -- defaults to the Kimi Coding Plan endpoint
  (`https://api.kimi.com/coding/v1`; use `https://api.kimi.ai/coding/v1` outside
  mainland China) when unset.
- `OPENAI_MODEL` -- defaults to `k3` when unset.

Any other OpenAI-compatible `/chat/completions` endpoint works too -- just point
`OPENAI_BASE_URL` / `OPENAI_MODEL` at it. On the platform, `OPENAI_BASE_URL` /
`OPENAI_API_KEY` are injected automatically for every run (the platform's own model
proxy and a temporary credential); `.env` is never uploaded and is excluded by
`pack_agent.py`.

## Packing and uploading as a complete project

```bash
python3 pack_agent.py --out ../python-agent.zip
```

This zips the project with `observer.project.json` at the root, skipping `.env`,
`__pycache__`, and other local-only files. Upload the resulting ZIP as a complete
project on the Participate page, or push this folder to a GitHub repository and
submit the repo URL instead.

## Protocol & scoring

Full protocol reference and scoring formulas are not duplicated here -- see `docs/`
(bundled in the downloadable ZIP of this example) for the complete participant
guide in Chinese and English.
