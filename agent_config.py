import json
from typing import List


TOOL_SCHEMAS: List[dict] = [
    {
        "name": "retrieve_from_kb",
        "description": "Retrieve relevant passages from the local MedQuad knowledge base.",
        "arguments": {
            "query": "string",
            "top_k": "integer, optional",
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a numerical expression for dosage, conversion, or arithmetic.",
        "arguments": {
            "expression": "string",
        },
    },
]


TOOL_DEFINITIONS_TEXT = "\n".join(
    f"- {schema['name']}({json.dumps(schema['arguments'], ensure_ascii=False)}) -> {schema['description']}"
    for schema in TOOL_SCHEMAS
)


AGENT_OUTPUT_FORMAT = """Assistant turn when a tool is needed:
Thought: <reason about the next step>
Action: <tool_name(arguments)>

Environment/tool turn:
<tool result returned by the runtime>

Assistant turn after receiving the tool result:
Thought: <reason using the tool result>
Final Answer: <final answer to the user>"""


AGENT_SYSTEM_PROMPT = f"""You are a medical assistant agent that can use local tools to answer user questions.

Available tools:
{TOOL_DEFINITIONS_TEXT}

Output format:
{AGENT_OUTPUT_FORMAT}

Rules:
- Think before acting.
- Use retrieve_from_kb for factual medical lookup grounded in the local knowledge base.
- Use calculator for arithmetic, dosage computation, or unit conversion.
- When you need a tool, stop after the Action line so the runtime can execute it.
- Never fabricate tool outputs or write an Observation yourself before the tool responds.
- If a tool is not needed, go directly to Final Answer.
- Keep the final answer factual, concise, and medically cautious."""
