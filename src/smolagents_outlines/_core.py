"""Outlines-constrained actions for native smolagents agents."""

from __future__ import annotations

import ast
import json
import keyword
import math
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, NoReturn, cast

import outlines
from jsonschema import Draft202012Validator, ValidationError
from outlines.inputs import Chat
from outlines.types import JsonSchema
from smolagents import Tool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
    get_clean_message_list,
    tool_role_conversions,
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase


_SCALAR_TYPES = {"string", "integer", "number", "boolean", "null"}
_SCHEMA_KEYS = {
    "type", "description", "enum", "const", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
    "pattern",
}
_CODE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"code": {"type": "string", "minLength": 1}},
    "required": ["code"],
    "additionalProperties": False,
}


def _schema_fingerprint(inputs: Mapping[str, Mapping[str, object]]) -> str:
    return json.dumps(inputs, sort_keys=True, allow_nan=False)


def _reject_nonfinite_constant(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON value is not a valid tool argument: {value}")


def _check_finite_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number is not a valid tool argument")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite_numbers(item)


def _python_identifier(name: str) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def _input_schema(
    tool_name: str, arg_name: str, spec: Mapping[str, object]
) -> dict[str, object]:
    unsupported = set(spec) - _SCHEMA_KEYS - {"nullable"}
    if unsupported:
        raise ValueError(
            f"{tool_name}.{arg_name} uses unsupported schema keys: {sorted(unsupported)}"
        )
    types = spec.get("type")
    types = [types] if isinstance(types, str) else types
    if (
        not isinstance(types, list)
        or not types
        or not all(isinstance(item, str) and item in _SCALAR_TYPES for item in types)
    ):
        raise ValueError(
            f"{tool_name}.{arg_name} needs JSON scalar input types; got {types!r}"
        )
    if spec.get("nullable"):
        types = list(dict.fromkeys([*types, "null"]))
    result = {key: value for key, value in spec.items() if key in _SCHEMA_KEYS}
    result["type"] = types[0] if len(types) == 1 else types
    return result


def _action_schema(
    tools: Sequence[Tool],
    final_answer_schema: dict[str, object] | None,
    final_answers: Sequence[str] | None,
) -> dict[str, object]:
    if not tools:
        raise ValueError("At least one custom tool is required")
    if final_answer_schema is not None and final_answers is not None:
        raise ValueError("Use final_answer_schema or final_answers, not both")
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)) or "final_answer" in names:
        raise ValueError("Tool names must be unique and must not include final_answer")
    branches: list[dict[str, object]] = []
    for tool in tools:
        if not _python_identifier(tool.name):
            raise ValueError(f"Tool name is not a Python identifier: {tool.name!r}")
        for arg_name in tool.inputs:
            if not _python_identifier(arg_name):
                raise ValueError(
                    f"Argument name is not a Python identifier: {tool.name}.{arg_name}"
                )
        properties: dict[str, dict[str, object]] = {
            key: _input_schema(tool.name, key, spec)
            for key, spec in tool.inputs.items()
        }
        branches.append({
            "type": "object",
            "properties": {
                "name": {"const": tool.name},
                "arguments": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    if final_answer_schema is not None:
        if not isinstance(final_answer_schema, dict):
            raise ValueError("final_answer_schema must be a JSON Schema object")
        answer = deepcopy(final_answer_schema)
        Draft202012Validator.check_schema(answer)
    elif final_answers is not None:
        if not final_answers or not all(isinstance(x, str) for x in final_answers):
            raise ValueError("final_answers must be a nonempty list of strings")
        answer = {"type": "string", "enum": list(dict.fromkeys(final_answers))}
    else:
        answer = {"type": "string"}
    branches.append({
        "type": "object",
        "properties": {
            "name": {"const": "final_answer"},
            "arguments": {
                "type": "object",
                "properties": {"answer": answer},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    })
    schema: dict[str, object] = {"oneOf": branches}
    Draft202012Validator.check_schema(schema)
    return schema


def _validate_code_calls(
    code: str, argument_schemas: Mapping[str, Mapping[str, object]]
) -> None:
    """Check Python syntax and direct calls to the declared tools."""
    tree = ast.parse(code)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in argument_schemas:
            parent = parents.get(node)
            if not isinstance(parent, ast.Call) or parent.func is not node:
                raise ValueError(f"Use {node.id} as a direct tool call")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in argument_schemas:
            continue
        schema = argument_schemas[name]
        properties = cast(dict[str, dict[str, object]], schema["properties"])
        arg_names = list(properties)
        if len(node.args) > len(arg_names) or any(isinstance(arg, ast.Starred) for arg in node.args):
            raise ValueError(f"{name} has invalid positional arguments")
        supplied = dict(zip(arg_names, node.args))
        for keyword_arg in node.keywords:
            key = keyword_arg.arg
            if key is None or key not in properties or key in supplied:
                raise ValueError(f"{name} has an unknown or duplicate keyword argument")
            supplied[key] = keyword_arg.value
        missing = set(cast(list[str], schema["required"])) - set(supplied)
        if missing:
            raise ValueError(f"{name} is missing arguments: {sorted(missing)}")
        for key, expression in supplied.items():
            try:
                value = ast.literal_eval(expression)
            except (ValueError, TypeError, RecursionError):
                continue  # A runtime expression can only be checked when it executes.
            _check_finite_numbers(value)
            Draft202012Validator(properties[key]).validate(value)


class OutlinesToolModel(Model):
    """Constrain native tool calls and check CodeAgent Python before execution."""

    def __init__(
        self,
        hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tools: Sequence[Tool],
        *,
        final_answer_schema: dict[str, object] | None = None,
        final_answers: Sequence[str] | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(model_id="outlines-transformers-smolagents")
        self.tools = list(tools)
        self.tool_inputs = {
            tool.name: _schema_fingerprint(tool.inputs) for tool in self.tools
        }
        self.schema = _action_schema(self.tools, final_answer_schema, final_answers)
        self.validator = Draft202012Validator(self.schema)
        self.code_validator = Draft202012Validator(_CODE_SCHEMA)
        self.argument_schemas = {
            cast(str, branch["properties"]["name"]["const"]): cast(
                dict[str, object], branch["properties"]["arguments"]
            )
            for branch in cast(list[dict[str, Any]], self.schema["oneOf"])
        }
        # Outlines' annotation omits fast tokenizers, which its adapter accepts.
        self.outlines_model = outlines.from_transformers(hf_model, tokenizer)  # type: ignore[arg-type]
        self.max_new_tokens = max_new_tokens

    def validate_tools(
        self, tools: Sequence[Tool], *, include_final_answer: bool = False
    ) -> None:
        runtime = {tool.name: tool for tool in tools}
        expected = set(self.tool_inputs)
        if include_final_answer:
            expected.add("final_answer")
        if set(runtime) != expected:
            raise RuntimeError("Agent tools differ from the constrained model's tools")
        for name, fingerprint in self.tool_inputs.items():
            if _schema_fingerprint(runtime[name].inputs) != fingerprint:
                raise RuntimeError(f"Agent tool schema changed: {name}")

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        if not stop_sequences or stop_sequences[:2] != ["Observation:", "Calling tools:"]:
            raise RuntimeError("Only agent action generation is supported")
        if response_format is not None:
            raise RuntimeError("CodeAgent structured-output mode is not supported")
        if kwargs:
            raise TypeError(f"Unsupported generation arguments: {sorted(kwargs)}")
        if tools_to_call_from is not None:
            self.validate_tools(tools_to_call_from, include_final_answer=True)
        self.validate_tools(self.tools)

        clean = get_clean_message_list(
            messages, role_conversions=tool_role_conversions,
            flatten_messages_as_text=True,
        )
        if tools_to_call_from is None:
            catalog = "\n".join(
                f"- {tool.name}: {tool.description}; inputs: "
                f"{json.dumps(tool.inputs, sort_keys=True)}"
                for tool in self.tools
            )
            system_prompt = (
                "Write Python for this CodeAgent step. You may use variables, "
                "arithmetic, conditionals, loops, and multiple tool calls. "
                "Call declared tools directly with explicit arguments. "
                "Call final_answer(answer=result) when ready; a bare Python "
                "expression does not finish the run. Available tools:\n"
                + catalog + "\n- final_answer: return the answer; arguments: "
                + json.dumps(self.argument_schemas["final_answer"])
            )
            if clean and clean[0]["role"] == "system":
                clean[0]["content"] += "\n\n" + system_prompt
            else:
                clean.insert(0, {"role": "system", "content": system_prompt})
            clean[-1]["content"] += (
                "\n\nReturn a JSON object with only a code field containing "
                "plain Python for this step. Do not include code tags or prose."
            )
            for attempt in range(3):
                raw = self.outlines_model(
                    Chat(clean), output_type=JsonSchema(_CODE_SCHEMA),
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                )
                try:
                    payload = cast(
                        dict[str, object],
                        json.loads(raw, parse_constant=_reject_nonfinite_constant),
                    )
                    self.code_validator.validate(payload)
                    code = cast(str, payload["code"])
                    if "<code>" in code or "</code>" in code:
                        raise ValueError("Do not put code-block tags inside code")
                    _validate_code_calls(code, self.argument_schemas)
                except (SyntaxError, ValueError, ValidationError) as exc:
                    if attempt == 2:
                        raise ValueError(f"CodeAgent code failed validation: {exc}") from exc
                    clean[-1]["content"] += (
                        f"\n\nPrevious code {raw!r} failed validation: {exc}. "
                        "Return corrected Python in the code field."
                    )
                    continue
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"<code>\n{code}\n</code>", raw=raw,
                )
            raise RuntimeError("CodeAgent code validation did not complete")
        clean[-1]["content"] += (
            "\n\nReturn exactly one JSON object with name and arguments for "
            "one available tool. This call will be executed as the next agent "
            "action. Use final_answer when the observations suffice. "
            "No explanation."
        )
        raw = self.outlines_model(
            Chat(clean), output_type=JsonSchema(self.schema),
            max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        action = cast(dict[str, object], json.loads(raw, parse_constant=_reject_nonfinite_constant))
        _check_finite_numbers(action)
        self.validator.validate(action)
        name = cast(str, action["name"])
        arguments = cast(dict[str, object], action["arguments"])
        return ChatMessage(
            role=MessageRole.ASSISTANT, content=raw, raw=raw,
            tool_calls=[ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=name, arguments=arguments,
                ),
                id=f"call_{uuid.uuid4().hex[:12]}", type="function",
            )],
        )
