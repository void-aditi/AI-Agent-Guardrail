"""
Verifies the LLM-driven graph's plumbing WITHOUT needing a real Ollama
server -- a fake LLM returns scripted responses so we can confirm:
  - tool_calls are correctly extracted into pending_action
  - interrupt()/resume correctly pauses and continues the graph
  - a BLOCKED result is fed back to the LLM as a ToolMessage
  - the LLM can react to that blocked result (propose something else)
  - the LLM finishing with plain text correctly ends the graph
  - the MAX_STEPS safety cap actually stops a runaway agent

Run: python test_agent_graph.py
"""

import shutil
from langchain_core.messages import AIMessage
from langgraph.types import Command

from interceptor.agent_graph import build_graph, initial_state, MAX_STEPS
from demo_agent.tools import seed_sandbox, SANDBOX_DIR


class ScriptedToolCallLLM:
    """Fake chat model: returns one canned AIMessage per .invoke() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):
        resp = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return resp


def make_tool_call_message(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def reset_sandbox():
    shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
    import os
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    seed_sandbox()


def run_graph_with_decisions(fake_llm, decisions, goal="Clean up the sandbox."):
    """Drives the graph exactly like the Flask app does, but decisions
    come from a pre-scripted list instead of a human clicking buttons."""
    app = build_graph(llm_with_tools=fake_llm)
    config = {"configurable": {"thread_id": "test-thread"}}
    result = app.invoke(initial_state(goal), config=config)

    decisions_iter = iter(decisions)
    while "__interrupt__" in result:
        decision = next(decisions_iter)
        result = app.invoke(Command(resume=decision), config=config)
    return result


def test_basic_tool_call_and_approval():
    print("TEST 1: single tool call, approved, LLM then finishes")
    reset_sandbox()
    fake_llm = ScriptedToolCallLLM([
        make_tool_call_message("read_file", {"filename": "report.txt"}, "call_1"),
        AIMessage(content="Done, I read the report."),  # no tool_calls -> finished
    ])
    result = run_graph_with_decisions(fake_llm, decisions=["allow"])
    assert len(result["decision_log"]) == 1, f"expected 1 log entry, got {len(result['decision_log'])}"
    assert result["decision_log"][0]["outcome"] == "EXECUTED"
    assert result["decision_log"][0]["tool"] == "read_file"
    assert result["finished"] is True
    print("  PASSED\n")


def test_block_result_feeds_back_to_llm():
    print("TEST 2: blocked action's result is visible to the LLM as a ToolMessage")
    reset_sandbox()
    fake_llm = ScriptedToolCallLLM([
        make_tool_call_message("delete_file", {"filename": "report.txt"}, "call_1"),
        AIMessage(content="Understood, I won't delete it."),
    ])
    result = run_graph_with_decisions(fake_llm, decisions=["block"])
    assert result["decision_log"][0]["outcome"] == "BLOCKED"
    # confirm a ToolMessage with the blocked result actually made it into history
    tool_msgs = [m for m in result["messages"] if m.__class__.__name__ == "ToolMessage"]
    assert len(tool_msgs) == 1
    assert "BLOCKED" in tool_msgs[0].content
    print("  PASSED (LLM received:", tool_msgs[0].content, ")\n")

    # also confirm the filesystem was genuinely untouched
    import os
    assert os.path.exists(os.path.join(SANDBOX_DIR, "report.txt")), "file should NOT have been deleted"
    print("  Filesystem verified untouched\n")


def test_multi_step_react_loop():
    print("TEST 3: multi-step loop -- LLM reacts across 3 tool calls")
    reset_sandbox()
    fake_llm = ScriptedToolCallLLM([
        make_tool_call_message("delete_file", {"filename": "scratch_notes.txt"}, "call_1"),
        make_tool_call_message("read_file", {"filename": "report.txt"}, "call_2"),
        make_tool_call_message("send_email", {"to": "a@b.com", "subject": "done", "body": "ok"}, "call_3"),
        AIMessage(content="All steps complete."),
    ])
    result = run_graph_with_decisions(fake_llm, decisions=["allow", "allow", "block"])
    assert len(result["decision_log"]) == 3
    outcomes = [e["outcome"] for e in result["decision_log"]]
    assert outcomes == ["EXECUTED", "EXECUTED", "BLOCKED"], outcomes
    print("  PASSED, outcomes:", outcomes, "\n")


def test_max_steps_safety_cap():
    print("TEST 4: MAX_STEPS safety cap stops a runaway agent")
    reset_sandbox()
    # an LLM that ALWAYS proposes another tool call, never stops on its own
    responses = [
        make_tool_call_message("read_file", {"filename": "report.txt"}, f"call_{i}")
        for i in range(MAX_STEPS + 5)
    ]
    fake_llm = ScriptedToolCallLLM(responses)
    decisions = ["allow"] * (MAX_STEPS + 5)
    result = run_graph_with_decisions(fake_llm, decisions=decisions)
    assert len(result["decision_log"]) == MAX_STEPS, (
        f"expected cap to stop at {MAX_STEPS}, got {len(result['decision_log'])}"
    )
    assert result["finished"] is True
    print(f"  PASSED, agent stopped itself at exactly {MAX_STEPS} steps despite LLM trying to continue\n")


if __name__ == "__main__":
    test_basic_tool_call_and_approval()
    test_block_result_feeds_back_to_llm()
    test_multi_step_react_loop()
    test_max_steps_safety_cap()
    print("ALL TESTS PASSED")
