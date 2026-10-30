"""Outlines-constrained actions for native smolagents agents."""

from __future__ import annotations

import json
import keyword
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import outlines
from jsonschema import Draft202012Validator
from outlines.inputs import Chat
from outlines.types import JsonSchema, Regex
from smolagents import Tool
from smolagents.local_python_executor import BASE_PYTHON_TOOLS, fix_final_answer_code
from smolagents.tools import validate_tool_arguments
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
    get_clean_message_list,
    tool_role_conversions,
)
from smolagents.utils import parse_code_blobs
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ._code_regex import build_code_regex
from ._code_static import validate_code_action


_SCALAR_TYPES = {"string", "integer", "number", "boolean", "null"}
_SCHEMA_KEYS = {
    "type", "description", "enum", "const", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
    "pattern",
}
_MAX_EXPANDED_INTEGER_CHOICES = 256
_CODE_BLOCK = re.compile(r"\A<code>\n(?P<code>[\s\S]*?)\n</code>\Z")


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


def _native_code_message(code: str) -> str:
    """Fail if native CodeAgent would parse or rewrite this generated block."""
    content = f"<code>\n{code}\n</code>"
    parsed = parse_code_blobs(content, ("<code>", "</code>"))
    if parsed != code or fix_final_answer_code(parsed) != code:
        raise ValueError("Native CodeAgent would change the generated code")
    return content


def _message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return "\n".join(
            part["text"] for part in message.content
            if isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def _prior_successful_code(messages: Sequence[ChatMessage]) -> list[str]:
    """Read successful CodeAgent blocks from this run's native message history."""
    successful: list[str] = []
    pending: str | None = None
    observed = False
    errored = False

    def finish_pending() -> None:
        nonlocal pending, observed, errored
        if pending is not None and observed and not errored:
            successful.append(pending)
        pending = None
        observed = False
        errored = False

    for message in messages:
        content = _message_text(message)
        if message.role == MessageRole.USER and content.startswith("New task:\n"):
            successful.clear()
            pending = None
            observed = False
            errored = False
        elif message.role == MessageRole.ASSISTANT:
            finish_pending()
            match = _CODE_BLOCK.fullmatch(content.strip())
            pending = match.group("code") if match is not None else None
        elif message.role == MessageRole.TOOL_RESPONSE and pending is not None:
            if content.startswith("Observation:\n"):
                observed = True
            else:
                errored = True
    finish_pending()
    return successful


def _python_identifier(name: str) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def _expand_finite_integer_choices(schema: dict[str, object]) -> dict[str, object]:
    """Represent small integer intervals as choices the pinned decoder enforces."""
    declared = schema.get("type")
    nullable_integer = (
        isinstance(declared, list)
        and all(isinstance(item, str) for item in declared)
        and set(declared) == {"integer", "null"}
    )
    if (
        (declared == "integer" or nullable_integer)
        and "enum" not in schema and "const" not in schema
    ):
        lower: int | None = None
        upper: int | None = None
        for key in ("minimum", "exclusiveMinimum"):
            value = schema.get(key)
            if type(value) is int or (type(value) is float and math.isfinite(value)):
                candidate = math.ceil(value) if key == "minimum" else math.floor(value) + 1
                lower = candidate if lower is None else max(lower, candidate)
        for key in ("maximum", "exclusiveMaximum"):
            value = schema.get(key)
            if type(value) is int or (type(value) is float and math.isfinite(value)):
                candidate = math.floor(value) if key == "maximum" else math.ceil(value) - 1
                upper = candidate if upper is None else min(upper, candidate)
        if lower is not None and upper is not None:
            if lower <= upper and upper - lower < _MAX_EXPANDED_INTEGER_CHOICES:
                schema["enum"] = list(range(lower, upper + 1)) + (
                    [None] if nullable_integer else []
                )
    for key in ("properties", "$defs", "patternProperties"):
        children = schema.get(key)
        if isinstance(children, dict):
            schema[key] = {
                name: _expand_finite_integer_choices(child)
                if isinstance(child, dict) else child
                for name, child in children.items()
            }
    for key in ("items", "additionalProperties", "not", "if", "then", "else"):
        child = schema.get(key)
        if isinstance(child, dict):
            schema[key] = _expand_finite_integer_choices(child)
    for key in ("oneOf", "anyOf", "allOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            schema[key] = [
                _expand_finite_integer_choices(child)
                if isinstance(child, dict) else child
                for child in children
            ]
    return schema


def _native_accepts_choice(spec: Mapping[str, object], value: object) -> bool:
    """Check the Python value that smolagents would pass to a scalar tool input."""
    schema_only_tool = cast(Tool, SimpleNamespace(inputs={"value": dict(spec)}))
    try:
        validate_tool_arguments(schema_only_tool, {"value": value})
    except (TypeError, ValueError):
        return False
    return True


def _constrain_discrete_input_choices(
    schema: dict[str, object], spec: Mapping[str, object], label: str
) -> dict[str, object]:
    """Keep decoder choices valid under both JSON Schema and native Python types."""
    if "enum" not in schema and "const" not in schema:
        return schema
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    invalid = object()

    def valid_choice(value: object) -> object:
        candidates = [value]
        if type(value) is float and math.isfinite(value) and value.is_integer():
            candidates.append(int(value))
        for candidate in candidates:
            try:
                _check_finite_numbers(candidate)
            except ValueError:
                continue
            if validator.is_valid(candidate) and _native_accepts_choice(spec, candidate):
                return candidate
        return invalid

    if "enum" in schema:
        choices: list[object] = []
        seen: set[str] = set()
        for original in cast(list[object], schema["enum"]):
            candidate = valid_choice(original)
            if candidate is invalid:
                continue
            identity = json.dumps(candidate, sort_keys=True, allow_nan=False)
            if identity not in seen:
                choices.append(candidate)
                seen.add(identity)
        if not choices:
            raise ValueError(f"{label} has no JSON- and Python-valid enum choices")
        schema["enum"] = choices
    if "const" in schema:
        candidate = valid_choice(schema["const"])
        if candidate is invalid:
            raise ValueError(f"{label} has no JSON- and Python-valid const value")
        schema["const"] = candidate
    return schema


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
    result = _constrain_discrete_input_choices(result, spec, f"{tool_name}.{arg_name}")
    # Outlines 1.3.3 ignores numeric bounds while building its JSON decoder.
    return _expand_finite_integer_choices(result)


def _action_schema(
    tools: Sequence[Tool],
    final_answer_schema: dict[str, object] | None,
    final_answers: Sequence[str] | None,
) -> dict[str, object]:
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
        answer = _expand_finite_integer_choices(deepcopy(final_answer_schema))
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


class OutlinesToolModel(Model):
    """Constrain native tool calls and a bounded CodeAgent Python subset."""

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
        self.tool_by_name = {tool.name: tool for tool in self.tools}
        self.tool_inputs = {
            tool.name: _schema_fingerprint(tool.inputs) for tool in self.tools
        }
        self.schema = _action_schema(self.tools, final_answer_schema, final_answers)
        self.validator = Draft202012Validator(self.schema)
        self.argument_schemas = {
            cast(str, branch["properties"]["name"]["const"]): cast(
                dict[str, object], branch["properties"]["arguments"]
            )
            for branch in cast(list[dict[str, Any]], self.schema["oneOf"])
        }
        self.final_answer_schema = cast(
            dict[str, object],
            cast(dict[str, Any], self.argument_schemas["final_answer"]["properties"])["answer"],
        )
        self.final_answer_validator = Draft202012Validator(self.final_answer_schema)
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
        if response_format is not None:
            raise RuntimeError("CodeAgent structured-output mode is not supported")
        if kwargs:
            raise TypeError(f"Unsupported generation arguments: {sorted(kwargs)}")
        is_action = bool(stop_sequences and stop_sequences[:2] == ["Observation:", "Calling tools:"])
        is_fallback = stop_sequences is None and tools_to_call_from is None
        if not is_action and not is_fallback:
            raise RuntimeError("Only agent actions and final-answer fallback are supported")
        if is_action and tools_to_call_from is not None:
            self.validate_tools(tools_to_call_from, include_final_answer=True)
        self.validate_tools(self.tools)

        clean = get_clean_message_list(
            messages, role_conversions=tool_role_conversions,
            flatten_messages_as_text=True,
        )
        if is_fallback:
            clean[-1]["content"] += (
                "\n\nReturn only a JSON value matching this final-answer schema: "
                + json.dumps(self.final_answer_schema, sort_keys=True)
            )
            raw = self.outlines_model(
                Chat(clean), output_type=JsonSchema(self.final_answer_schema),
                max_new_tokens=self.max_new_tokens, do_sample=False,
            )
            answer = json.loads(raw, parse_constant=_reject_nonfinite_constant)
            _check_finite_numbers(answer)
            self.final_answer_validator.validate(answer)
            content = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            return ChatMessage(role=MessageRole.ASSISTANT, content=content, raw=raw)
        if tools_to_call_from is None:
            collisions = set(self.tool_by_name).intersection(BASE_PYTHON_TOOLS)
            if collisions:
                raise RuntimeError(
                    "CodeAgent tool names conflict with native Python functions: "
                    + ", ".join(sorted(collisions))
                )
            catalog = "\n".join(
                f"- {tool.name}: {tool.description}; inputs: "
                f"{json.dumps(tool.inputs, sort_keys=True)}"
                for tool in self.tools
            )
            system_prompt = (
                "Write straight-line Python for this CodeAgent step. You may "
                "assign v0, v1, and other v-numbered variables and chain "
                "multiple direct tool calls. Each tool argument must be a "
                "v-numbered variable or a literal shaped like its declared "
                "type. Assign a variable earlier in this run before "
                "using it; unknown tool outputs can only feed plain typed "
                "arguments. Do not use loops, conditionals, imports, nested calls, "
                "or bare expressions. Call final_answer(answer=...) when ready. "
                "Available tools:\n"
                + catalog + "\n- final_answer: return the answer; arguments: "
                + json.dumps(self.argument_schemas["final_answer"])
            )
            if clean and clean[0]["role"] == "system":
                clean[0]["content"] += "\n\n" + system_prompt
            else:
                clean.insert(0, {"role": "system", "content": system_prompt})
            clean[-1]["content"] += (
                "\n\nReturn only the Python statements for this step, without "
                "code tags or prose. Use v-numbered variables for intermediate "
                "results and explicit keyword arguments for tool calls."
            )
            pattern = build_code_regex(self.schema)
            raw = self.outlines_model(
                Chat(clean), output_type=Regex(pattern),
                max_new_tokens=self.max_new_tokens, do_sample=False,
            )
            if re.fullmatch(pattern, raw) is None:
                raise ValueError("CodeAgent generation ended outside the Python constraint")
            validate_code_action(
                raw, self.argument_schemas, self.tool_by_name,
                prior_successful_code=_prior_successful_code(messages),
            )
            content = _native_code_message(raw)
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content, raw=raw,
            )
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
        if name != "final_answer":
            validate_tool_arguments(self.tool_by_name[name], arguments)
        return ChatMessage(
            role=MessageRole.ASSISTANT, content=raw, raw=raw,
            tool_calls=[ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=name, arguments=arguments,
                ),
                id=f"call_{uuid.uuid4().hex[:12]}", type="function",
            )],
        )
