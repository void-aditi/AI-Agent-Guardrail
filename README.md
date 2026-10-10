# AI-Agent-Guardrail
BE Finaleee year Project | ML-based AI agent security framework with runtime guardrails, risk scoring, human approval, and blockchain audit trails.
guardrail-phase1/
├── main.py                 # run this — drives the demo + console prompts
├── demo_agent/
│   ├── tools.py             # the 3 tools an "agent" can call (1 safe, 2 risky)
│   └── sandbox/              # a throwaway folder the tools operate on
└── interceptor/
    └── graph.py              # the LangGraph graph + interception checkpoint