"""
LLM-driven guardrail demo backend.

Same interception guarantees as the Phase 1 visual demo -- the only
difference is that the agent's proposed actions now come from a real
LLM (via Ollama) reasoning over a goal you type in, instead of a
hardcoded script.

Requires Ollama running locally with a tool-calling-capable model
pulled (see README.md). If Ollama isn't reachable, /api/start returns
a clear error instead of crashing.
"""

import os
import shutil
import uuid
from flask import Flask, jsonify, request, send_from_directory

from langgraph.types import Command
from interceptor.agent_graph import build_graph, initial_state
from demo_agent.tools import seed_sandbox, SANDBOX_DIR

app = Flask(__name__, static_folder="static", static_url_path="")

SESSION = {"result": None, "thread_id": None, "graph": None}


def fresh_sandbox():
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    seed_sandbox()


def build_status():
    result = SESSION["result"]
    pending = None
    done = True
    if result is not None and "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        pending = {"tool": payload["tool"], "args": payload["args"]}
        done = False
    final_message = None
    if result is not None and done:
        # last AI message (if any) is the LLM's closing explanation
        ai_msgs = [m for m in result.get("messages", []) if m.__class__.__name__ == "AIMessage" and getattr(m, "content", "")]
        if ai_msgs:
            final_message = ai_msgs[-1].content
    return {
        "pending": pending,
        "done": done,
        "log": result.get("decision_log", []) if result else [],
        "final_message": final_message,
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/start", methods=["POST"])
def start():
    goal = (request.json or {}).get("goal") or (
        "Review the sandbox files. Read report.txt. Then decide whether "
        "scratch_notes.txt should be deleted, and email a summary to "
        "finance-team@example.com."
    )

    if SESSION["graph"] is None:
        try:
            SESSION["graph"] = build_graph()  # real ChatOllama, uses env vars
        except Exception as e:
            return jsonify({"error": f"Could not reach Ollama: {e}"}), 503

    fresh_sandbox()
    SESSION["thread_id"] = str(uuid.uuid4())
    config = {"configurable": {"thread_id": SESSION["thread_id"]}}

    try:
        SESSION["result"] = SESSION["graph"].invoke(initial_state(goal), config=config)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}. Is 'ollama serve' running and the model pulled?"}), 503

    return jsonify(build_status())


@app.route("/api/decide", methods=["POST"])
def decide():
    decision = (request.json or {}).get("decision")
    if decision not in ("allow", "block"):
        return jsonify({"error": "decision must be 'allow' or 'block'"}), 400
    config = {"configurable": {"thread_id": SESSION["thread_id"]}}
    try:
        SESSION["result"] = SESSION["graph"].invoke(Command(resume=decision), config=config)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 503
    return jsonify(build_status())


@app.route("/api/sandbox")
def sandbox():
    files = {}
    if os.path.exists(SANDBOX_DIR):
        for fname in sorted(os.listdir(SANDBOX_DIR)):
            with open(os.path.join(SANDBOX_DIR, fname)) as f:
                files[fname] = f.read()
    return jsonify(files)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
