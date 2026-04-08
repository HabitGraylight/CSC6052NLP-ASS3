import ast
import json
import operator
import re
from typing import Any, Tuple

from agent_config import AGENT_OUTPUT_FORMAT, AGENT_SYSTEM_PROMPT, TOOL_DEFINITIONS_TEXT, TOOL_SCHEMAS
from vanilla_rag import retrieve as rag_retrieve


ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
        return ALLOWED_UNARY_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
        args = [_safe_eval(arg) for arg in node.args]
        return ALLOWED_FUNCTIONS[node.func.id](*args)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    parsed = ast.parse(expression, mode="eval")
    result = _safe_eval(parsed)
    if isinstance(result, float):
        text = f"{result:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(result)


def retrieve_from_kb(query: str, top_k: int = 3) -> str:
    return rag_retrieve(query=query, top_k=top_k)


def parse_action(text: str) -> Tuple[str | None, Any]:
    match = re.search(r"Action:\s*(\w+)\((.*)\)\s*$", text.strip(), flags=re.DOTALL)
    if not match:
        return None, None

    tool_name = match.group(1)
    raw_args = match.group(2).strip()
    if not raw_args:
        return tool_name, ()

    try:
        parsed_args = ast.literal_eval(raw_args)
    except Exception:
        parsed_args = raw_args
    return tool_name, parsed_args


def execute_tool(tool_name: str, args: Any) -> str:
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {sorted(TOOL_REGISTRY)}"

    tool = TOOL_REGISTRY[tool_name]
    try:
        if isinstance(args, dict):
            return tool(**args)
        if isinstance(args, (list, tuple)):
            return tool(*args)
        return tool(args)
    except Exception as exc:
        return f"Error executing {tool_name}: {exc}"


def tool_schemas_json() -> str:
    return json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)


TOOL_REGISTRY = {
    "retrieve_from_kb": retrieve_from_kb,
    "calculator": calculator,
}


__all__ = [
    "AGENT_OUTPUT_FORMAT",
    "AGENT_SYSTEM_PROMPT",
    "TOOL_DEFINITIONS_TEXT",
    "TOOL_SCHEMAS",
    "TOOL_REGISTRY",
    "calculator",
    "execute_tool",
    "parse_action",
    "retrieve_from_kb",
    "tool_schemas_json",
]
