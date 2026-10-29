"""Outlines-constrained tool actions for smolagents 1.26.0.

The model is the public integration point. It returns a native tool call to
ToolCallingAgent or a single Python invocation to CodeAgent.
"""

from __future__ import annotations

import json
import importlib.resources
import keyword
import math
import uuid
from copy import deepcopy
from typing import Any

import outlines
import yaml
from jsonschema import Draft202012Validator
from outlines.inputs import Chat
from outlines.types import JsonSchema
from smolagents import CodeAgent, ToolCallingAgent
from smolagents.memory import ActionStep
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
    get_clean_message_list,
    tool_role_conversions,
)


_SCALAR_TYPES = {"string", "integer", "number", "boolean", "null"}
_SCHEMA_KEYS = {
    "type", "description", "enum", "const", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
    "pattern",
}


def _schema_fingerprint(inputs: dict) -> str:
    return json.dumps(inputs, sort_keys=True, allow_nan=False)


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"Nonfinite JSON value is not a valid tool argument: {value}")


def _check_finite_numbers(value: Any) -> None:
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


def _input_schema(tool_name: str, arg_name: str, spec: dict) -> dict:
    unsupported = set(spec) - _SCHEMA_KEYS - {"nullable"}
    if unsupported:
        raise ValueError(
            f"{tool_name}.{arg_name} uses unsupported schema keys: {sorted(unsupported)}"
        )
    types = spec.get("type")
    types = [types] if isinstance(types, str) else types
    if not isinstance(types, list) or not types or not set(types) <= _SCALAR_TYPES:
        raise ValueError(
            f"{tool_name}.{arg_name} needs JSON scalar input types; got {types!r}"
        )
    if spec.get("nullable"):
        types = list(dict.fromkeys([*types, "null"]))
    result = {key: value for key, value in spec.items() if key in _SCHEMA_KEYS}
    result["type"] = types[0] if len(types) == 1 else types
    return result


def _action_schema(
    tools: list,
    final_answer_schema: dict | None,
    final_answers: list[str] | None,
) -> dict:
    if not tools:
        raise ValueError("At least one custom tool is required")
    if final_answer_schema is not None and final_answers is not None:
        raise ValueError("Use final_answer_schema or final_answers, not both")
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)) or "final_answer" in names:
        raise ValueError("Tool names must be unique and must not include final_answer")
    branches = []
    for tool in tools:
        if not _python_identifier(tool.name):
            raise ValueError(f"Tool name is not a Python identifier: {tool.name!r}")
        for arg_name in tool.inputs:
            if not _python_identifier(arg_name):
                raise ValueError(
                    f"Argument name is not a Python identifier: {tool.name}.{arg_name}"
                )
        properties = {
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
    schema = {"oneOf": branches}
    Draft202012Validator.check_schema(schema)
    return schema


class OutlinesToolModel(Model):
    """A smolagents model that emits only schema-valid action calls."""

    def __init__(
        self,
        hf_model: Any,
        tokenizer: Any,
        tools: list,
        *,
        final_answer_schema: dict | None = None,
        final_answers: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(model_id="outlines-transformers-smolagents")
        self.tools = list(tools)
        self.tool_inputs = {
            tool.name: _schema_fingerprint(tool.inputs) for tool in self.tools
        }
        self.schema = _action_schema(self.tools, final_answer_schema, final_answers)
        self.validator = Draft202012Validator(self.schema)
        self.outlines_model = outlines.from_transformers(hf_model, tokenizer)
        self.max_new_tokens = max_new_tokens

    def validate_tools(self, tools: list, *, include_final_answer: bool = False) -> None:
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
        self, messages, stop_sequences=None, tools_to_call_from=None, **kwargs
    ) -> ChatMessage:
        if not stop_sequences or stop_sequences[:2] != ["Observation:", "Calling tools:"]:
            raise RuntimeError("Only agent action generation is supported")
        if kwargs.get("response_format") is not None:
            raise RuntimeError("CodeAgent structured-output mode is not supported")
        if tools_to_call_from is not None:
            self.validate_tools(tools_to_call_from, include_final_answer=True)
        self.validate_tools(self.tools)

        clean = get_clean_message_list(
            messages, role_conversions=tool_role_conversions,
            flatten_messages_as_text=True,
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
        action = json.loads(raw, parse_constant=_reject_nonfinite_constant)
        _check_finite_numbers(action)
        self.validator.validate(action)
        name, arguments = action["name"], action["arguments"]
        if tools_to_call_from is None:
            code = name + "(" + ", ".join(
                f"{key}={value!r}" for key, value in arguments.items()
            ) + ")"
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"<code>\n{code}\n</code>", raw=raw,
            )
        return ChatMessage(
            role=MessageRole.ASSISTANT, content=raw, raw=raw,
            tool_calls=[ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=name, arguments=arguments,
                ),
                id=f"call_{uuid.uuid4().hex[:12]}", type="function",
            )],
        )


def _stop_on_step_error(step, agent) -> None:
    if isinstance(step, ActionStep) and step.error is not None:
        raise RuntimeError(f"Agent step failed: {step.error}")


class _FailClosed:
    def _handle_max_steps_reached(self, task):
        raise RuntimeError("Step limit reached without a final answer")


def _agent_kwargs(model: OutlinesToolModel, tools: list, kwargs: dict) -> dict:
    if not isinstance(model, OutlinesToolModel):
        raise TypeError("model must be an OutlinesToolModel")
    model.validate_tools(tools)
    if kwargs.get("managed_agents"):
        raise ValueError("Managed agents are outside this tool schema")
    if kwargs.get("add_base_tools"):
        raise ValueError("Additional base tools are outside this tool schema")
    if kwargs.get("planning_interval") is not None:
        raise ValueError("Planning generation is outside this action schema")
    if kwargs.get("stream_outputs"):
        raise ValueError("Streaming model output is not supported")
    callbacks = kwargs.pop("step_callbacks", None)
    if callbacks is None:
        callbacks = []
    if not isinstance(callbacks, list):
        raise TypeError("step_callbacks must be a list")
    kwargs["step_callbacks"] = [_stop_on_step_error, *callbacks]
    return kwargs


class OutlinesToolCallingAgent(_FailClosed, ToolCallingAgent):
    """ToolCallingAgent integration with explicit step and limit failures."""

    def __init__(self, tools: list, model: OutlinesToolModel, **kwargs):
        super().__init__(tools=tools, model=model, **_agent_kwargs(model, tools, kwargs))


class OutlinesCodeAgent(_FailClosed, CodeAgent):
    """CodeAgent integration restricted to one tool invocation per step."""

    def __init__(self, tools: list, model: OutlinesToolModel, **kwargs):
        if kwargs.get("use_structured_outputs_internally"):
            raise ValueError("CodeAgent structured code mode is not supported")
        if kwargs.get("code_block_tags") is not None:
            raise ValueError("CodeAgent requires its default <code> tags")
        if kwargs.get("prompt_templates") is None:
            kwargs["prompt_templates"] = yaml.safe_load(
                importlib.resources.files("smolagents.prompts")
                .joinpath("toolcalling_agent.yaml").read_text()
            )
        super().__init__(tools=tools, model=model, **_agent_kwargs(model, tools, kwargs))

    def _step_stream(self, memory_step):
        self.model.validate_tools([
            tool for name, tool in self.tools.items() if name != "final_answer"
        ])
        yield from super()._step_stream(memory_step)
