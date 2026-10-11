"""
LLM-driven guardrail graph.

Same core principle as the Phase 1 scripted version: every tool call is
intercepted and paused before execution. The difference is WHO decides
what action to propose next -- here it's a real LLM (via Ollama)
reasoning step by step, instead of a hardcoded list.

Flow:
    llm_propose_action  -> the LLM looks at the goal + history and either
                            (a) proposes a tool call, or
                            (b) decides it's done and responds in text
    interceptor         -> if (a): pause via interrupt(), wait for a
                            human allow/block decision
    execute_action       -> run (or skip) the tool, feed the result back
                            into the LLM's message history as a ToolMessage
    [loop back to llm_propose_action, so the LLM can react to the result]

A max_steps cap is included as an extra safety rail so a live demo
can't accidentally run away in an infinite tool-calling loop.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from demo_agent.tools import TOOL_REGISTRY, LLM_TOOLS

MAX_STEPS = 6  # hard cap so a live demo can't loop forever

SYSTEM_PROMPT = """You are an AI agent with access to tools that act on a user's files and email.
Every tool call you propose will be reviewed by a human guardrail before it executes -- you will
be told the outcome (executed or blocked) and should continue working toward the user's goal.
When you believe the goal is fully handled, respond in plain text with no further tool call."""


class GuardrailState(TypedDict):
    messages: List[Any]                  # LangChain message history (LLM's context)
    pending_action: Dict[str, Any]        # tool name + args + tool_call id awaiting a decision
    decision_log: List[Dict[str, Any]]    # full history: action + decision + result
    last_decision: str
    finished: bool
    steps: int


def make_llm_propose_action(llm_with_tools):
    """
    llm_with_tools is injected so this graph can be tested with a fake,
    scripted LLM (see test_agent_graph.py) without needing a real Ollama
    server running. In production, app.py passes a real ChatOllama
    instance bound to the tool schemas.
    """

    def llm_propose_action(state: GuardrailState) -> GuardrailState:
        if state["steps"] >= MAX_STEPS:
            return {**state, "finished": True}

        response = llm_with_tools.invoke(state["messages"])
        new_messages = state["messages"] + [response]

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # LLM chose not to call a tool -- it considers itself done.
            return {**state, "messages": new_messages, "finished": True}

        call = tool_calls[0]  # one action at a time, so each gets its own approval
        pending = {"tool": call["name"], "args": call["args"], "id": call["id"]}
        return {
            **state,
            "messages": new_messages,
            "pending_action": pending,
            "finished": False,
            "steps": state["steps"] + 1,
        }

    return llm_propose_action


def interceptor(state: GuardrailState) -> Command:
    """Identical safety checkpoint to Phase 1 -- every action pauses here."""
    action = state["pending_action"]
    decision = interrupt(
        {
            "event": "action_intercepted",
            "tool": action["tool"],
            "args": action["args"],
        }
    )
    return Command(update={"last_decision": decision}, goto="execute_action")


def execute_action(state: GuardrailState) -> GuardrailState:
    action = state["pending_action"]
    decision = state.get("last_decision", "block")  # fail-closed by default
    tool_name = action["tool"]
    tool_entry = TOOL_REGISTRY[tool_name]

    if decision == "allow":
        result = tool_entry["fn"](**action["args"])
        outcome = "EXECUTED"
    else:
        result = f"[BLOCKED] '{tool_name}' was not executed by the guardrail."
        outcome = "BLOCKED"

    log_entry = {
        "tool": tool_name,
        "args": action["args"],
        "decision": decision,
        "outcome": outcome,
        "result": result,
    }

    # Feed the real-world outcome back to the LLM so it can react --
    # e.g. if blocked, it should not simply retry the same action.
    tool_message = ToolMessage(content=str(result), tool_call_id=action["id"])

    return {
        **state,
        "decision_log": state["decision_log"] + [log_entry],
        "messages": state["messages"] + [tool_message],
    }


def route_after_propose(state: GuardrailState) -> str:
    return END if state["finished"] else "interceptor"


def route_after_execution(state: GuardrailState) -> str:
    return "llm_propose_action"


def build_graph(llm_with_tools=None, ollama_model: Optional[str] = None, ollama_base_url: Optional[str] = None):
    """
    Pass llm_with_tools explicitly for testing (a fake/mock object).
    In production, leave it None and a real ChatOllama is constructed
    from ollama_model / ollama_base_url (or environment variables).
    """
    if llm_with_tools is None:
        import os
        from langchain_ollama import ChatOllama
        model = ollama_model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        llm_with_tools = llm.bind_tools(LLM_TOOLS)

    graph = StateGraph(GuardrailState)
    graph.add_node("llm_propose_action", make_llm_propose_action(llm_with_tools))
    graph.add_node("interceptor", interceptor)
    graph.add_node("execute_action", execute_action)

    graph.add_edge(START, "llm_propose_action")
    graph.add_conditional_edges(
        "llm_propose_action",
        route_after_propose,
        {"interceptor": "interceptor", END: END},
    )
    # "interceptor" routes to "execute_action" itself via Command(goto=...)
    graph.add_conditional_edges(
        "execute_action",
        route_after_execution,
        {"llm_propose_action": "llm_propose_action"},
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def initial_state(goal: str) -> GuardrailState:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=goal)],
        "pending_action": {},
        "decision_log": [],
        "last_decision": "",
        "finished": False,
        "steps": 0,
    }
