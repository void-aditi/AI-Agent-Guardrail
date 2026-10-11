"""
Verifies the Flask API itself (routes, JSON shapes, status codes) by
injecting the same mocked LLM used in test_agent_graph.py directly into
the app's session -- so this checks the HTTP layer without needing a
live Ollama server.

Run: python test_app.py
"""

from langchain_core.messages import AIMessage

import app as app_module
from interceptor.agent_graph import build_graph


class ScriptedToolCallLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):
        resp = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return resp


def make_tool_call_message(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def main():
    fake_llm = ScriptedToolCallLLM([
        make_tool_call_message("read_file", {"filename": "report.txt"}, "call_1"),
        make_tool_call_message("delete_file", {"filename": "scratch_notes.txt"}, "call_2"),
        AIMessage(content="I've reviewed the files and handled the cleanup."),
    ])

    # Inject the mock graph directly so /api/start skips real Ollama construction
    app_module.SESSION["graph"] = build_graph(llm_with_tools=fake_llm)

    client = app_module.app.test_client()

    print("=== POST /api/start ===")
    r = client.post("/api/start", json={"goal": "Clean up the sandbox."})
    print(r.status_code, r.get_json())
    assert r.status_code == 200
    data = r.get_json()
    assert data["pending"]["tool"] == "read_file"
    assert data["done"] is False

    print("\n=== POST /api/decide allow ===")
    r = client.post("/api/decide", json={"decision": "allow"})
    data = r.get_json()
    print(r.status_code, data)
    assert data["pending"]["tool"] == "delete_file"
    assert data["log"][0]["outcome"] == "EXECUTED"

    print("\n=== POST /api/decide block ===")
    r = client.post("/api/decide", json={"decision": "block"})
    data = r.get_json()
    print(r.status_code, data)
    assert data["done"] is True
    assert data["log"][1]["outcome"] == "BLOCKED"
    assert data["final_message"] == "I've reviewed the files and handled the cleanup."

    print("\n=== GET /api/sandbox ===")
    r = client.get("/api/sandbox")
    data = r.get_json()
    print(r.status_code, data)
    assert "report.txt" in data
    assert "scratch_notes.txt" in data  # delete was blocked, must still exist

    print("\n=== POST /api/decide with bad value (should 400) ===")
    r = client.post("/api/decide", json={"decision": "maybe"})
    print(r.status_code, r.get_json())
    assert r.status_code == 400

    print("\nALL APP TESTS PASSED")


if __name__ == "__main__":
    main()
