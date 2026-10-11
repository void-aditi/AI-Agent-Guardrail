# AI Agent Action Guardrail — LLM-Driven Demo

This upgrades the visual demo so a **real local LLM (via Ollama)**
decides what action to take next, instead of a hardcoded script. The
interception guarantee is identical to before: every proposed tool
call is paused and requires your explicit Allow/Block before it
executes — now you're approving a real model's real decisions.

## ⚠️ What's tested vs. what you need to verify yourself

I built and thoroughly tested this, but **could not run a real Ollama
model** in the environment I built it in (no internet access to
download model weights there). Here's exactly what was verified and
what wasn't:

**✅ Tested (via `test_agent_graph.py` and `test_app.py`, using a
scripted fake LLM standing in for Ollama):**
- Tool calls are correctly extracted and routed to the interceptor
- `interrupt()`/resume correctly pauses and resumes the graph
- A BLOCKED result is correctly fed back to the LLM as a `ToolMessage`
  so it can react (confirmed the model would see it)
- A multi-step loop where the agent reacts across 3 chained actions
- The `MAX_STEPS` safety cap genuinely stops a runaway agent (tested
  with a fake LLM that never stops on its own — it was cut off at
  exactly 6 steps)
- The full Flask API contract (`/api/start`, `/api/decide`,
  `/api/sandbox`), including error responses
- Blocked deletes leave the real sandbox files untouched

**❌ NOT tested (do this yourself before the sponsor meeting):**
- Whether your installed Ollama model actually calls tools correctly
  in response to real prompts (tool-calling quality varies a lot by
  model — see recommendation below)
- Real end-to-end latency (local LLM inference is slower than the
  mocked tests — budget a few seconds per step)
- **Run `python test_app.py` again after setup — it'll still pass
  since it uses the mock, but then do a real manual run through the
  browser before presenting, to catch anything model-specific.**

## Setup

**1. Install Ollama** (one-time, on your machine):
https://ollama.com/download

**2. Pull a tool-calling-capable model.** Not all local models support
function/tool calling reliably. Recommended, in order of quality vs. size:

```bash
ollama pull llama3.1        # 8B, best balance of quality + speed, recommended
# or, if your machine is slower:
ollama pull qwen2.5:7b       # also solid tool-calling support, slightly smaller
```

**3. Start the Ollama server** (keep this running in its own terminal):

```bash
ollama serve
```

**4. Install Python dependencies:**

```bash
cd guardrail-llm
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**5. (Optional) Point at a different model/host:**

```bash
export OLLAMA_MODEL=qwen2.5:7b          # default: llama3.1
export OLLAMA_BASE_URL=http://localhost:11434   # default shown
```

## Run it

```bash
python app.py
```

Open **http://127.0.0.1:5000**, type a goal (or use the pre-filled
default), click **Start Agent**, and approve/block each action it
proposes as it reasons through the task.

## Run the tests (no Ollama needed)

```bash
python test_agent_graph.py   # graph logic, via a scripted fake LLM
python test_app.py           # Flask API, via the same fake LLM
```

Both should print `ALL TESTS PASSED` without Ollama running at all —
they verify the plumbing is correct independent of the model.

## What's different from the Phase 1 visual demo

| | Phase 1 (scripted) | This version (LLM-driven) |
|---|---|---|
| What decides the next action | Hardcoded list of 4 actions | A real LLM reasoning over your typed goal |
| Can react to a blocked action | No (fixed script) | Yes — sees the BLOCKED result and can change plan |
| Safety cap | N/A (fixed length) | `MAX_STEPS = 6`, tested to actually stop a looping agent |
| Interception mechanism | Same `interrupt()` pattern | Unchanged |

## Troubleshooting

- **"Could not reach Ollama" / Connection refused** → `ollama serve`
  isn't running, or is on a different port than `OLLAMA_BASE_URL`
- **Agent never proposes a tool call, just talks** → the model you
  pulled may not support tool calling well; try `llama3.1` specifically
- **Agent seems to ignore a BLOCKED result and retries the same thing**
  → this is a real model-quality limitation, not a bug in the
  interceptor — worth mentioning as an honest limitation if it comes
  up live, and a good discussion point for Phase 2's Risk Fusion layer
  eventually overriding repeated attempts automatically
