"""
Tools the LLM agent can choose to call.

Each tool is defined twice on purpose:
  1. A plain Python function (the actual execution logic) -- this is
     what the interceptor calls AFTER a decision is made.
  2. A LangChain @tool-wrapped version with a docstring + type hints --
     this is what gets shown to the LLM so it can decide to call it.

They are kept separate deliberately: the LLM only ever sees the tool
*schema* (name, description, arguments) and proposes a call. It never
directly executes anything -- that always goes through the interceptor
first. This separation is the whole point of the guardrail.
"""

import os
from langchain_core.tools import tool

SANDBOX_DIR = os.path.join(os.path.dirname(__file__), "sandbox")
os.makedirs(SANDBOX_DIR, exist_ok=True)


# ---------- actual execution logic (called only after a decision) ----------

def _read_file(filename: str) -> str:
    path = os.path.join(SANDBOX_DIR, filename)
    if not os.path.exists(path):
        return f"'{filename}' does not exist in sandbox."
    with open(path, "r") as f:
        content = f.read()
    return f"Contents of '{filename}':\n{content}"


def _delete_file(filename: str) -> str:
    path = os.path.join(SANDBOX_DIR, filename)
    if not os.path.exists(path):
        return f"'{filename}' does not exist in sandbox, nothing to delete."
    os.remove(path)
    return f"'{filename}' has been permanently deleted."


def _send_email(to: str, subject: str, body: str) -> str:
    # Simulated only -- no real email is sent.
    return f"Email sent to {to} | subject='{subject}'"


TOOL_REGISTRY = {
    "read_file": {"fn": _read_file, "risky": False},
    "delete_file": {"fn": _delete_file, "risky": True},
    "send_email": {"fn": _send_email, "risky": True},
}


# ---------- schemas the LLM sees (used only for deciding, never executing) ----------

@tool
def read_file(filename: str) -> str:
    """Read the contents of a file in the sandbox. Safe, read-only action."""
    return "[preview only -- real execution happens after guardrail approval]"


@tool
def delete_file(filename: str) -> str:
    """Permanently delete a file from the sandbox. This is irreversible."""
    return "[preview only -- real execution happens after guardrail approval]"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email on the user's behalf. This has a real external effect."""
    return "[preview only -- real execution happens after guardrail approval]"


LLM_TOOLS = [read_file, delete_file, send_email]


def seed_sandbox():
    """Create a couple of sample files so the agent has something to act on."""
    p1 = os.path.join(SANDBOX_DIR, "report.txt")
    if not os.path.exists(p1):
        with open(p1, "w") as f:
            f.write("Q3 financial summary -- do not delete without review.\n")
    p2 = os.path.join(SANDBOX_DIR, "scratch_notes.txt")
    if not os.path.exists(p2):
        with open(p2, "w") as f:
            f.write("Just some scratch notes, safe to delete.\n")
