"""Contract tests for generated actions and agent integration."""

import ast
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError
from smolagents import CodeAgent, Tool, ToolCallingAgent
from smolagents.models import ChatMessage, MessageRole
from smolagents.utils import AgentGenerationError

from smolagents_outlines import OutlinesToolModel


class EchoTool(Tool):
    name = "echo"
    description = "Return the supplied text."
    inputs = {"value": {"type": "string", "description": "Text to echo"}}
    output_type = "string"

    def forward(self, value: str) -> str:
        return value


class FakeOutlines:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.inputs = []

    def __call__(self, *args, **kwargs):
        self.inputs.append(args[0])
        return next(self.responses)


@pytest.fixture
def make_model(monkeypatch):
    def build(*responses, **kwargs):
        fake = FakeOutlines(*responses)
        monkeypatch.setattr(
            "smolagents_outlines._core.outlines.from_transformers",
            lambda *_: fake,
        )
        tool = EchoTool()
        tool.inputs = deepcopy(tool.inputs)
        return OutlinesToolModel(None, None, [tool], **kwargs), tool

    return build


def _messages():
    return [ChatMessage(
        role=MessageRole.USER,
        content=[{"type": "text", "text": "Choose an action"}],
    )]


def _generate(model, tools=None):
    return model.generate(
        _messages(),
        stop_sequences=["Observation:", "Calling tools:"],
        tools_to_call_from=tools,
    )


def _runtime_tools(tool):
    return [tool, SimpleNamespace(name="final_answer")]


def test_free_form_final_answer_is_default(make_model):
    answer = "Any explanation that fits this task"
    model, tool = make_model(json.dumps({
        "name": "final_answer", "arguments": {"answer": answer},
    }))
    message = _generate(model, _runtime_tools(tool))
    assert message.tool_calls[0].function.name == "final_answer"
    assert message.tool_calls[0].function.arguments == {"answer": answer}


def test_restricted_and_structured_final_answers(make_model):
    invalid = json.dumps({"name": "final_answer", "arguments": {"answer": "different"}})
    model, tool = make_model(invalid, final_answers=["Compact", "Standard"])
    with pytest.raises(ValidationError):
        _generate(model, _runtime_tools(tool))

    structured = json.dumps({
        "name": "final_answer", "arguments": {"answer": {"choice": "Standard"}},
    })
    model, _ = make_model(
        structured,
        final_answer_schema={
            "type": "object",
            "properties": {"choice": {"type": "string"}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    )
    message = _generate(model)
    assert "final_answer(answer={'choice': 'Standard'})" in message.content
    ast.parse(message.content.removeprefix("<code>\n").removesuffix("\n</code>"))

    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string"}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    for agent_cls in (ToolCallingAgent, CodeAgent):
        model, tool = make_model(structured, final_answer_schema=schema)
        agent = agent_cls(tools=[tool], model=model, max_steps=2)
        assert agent.run("Choose") == {"choice": "Standard"}


def test_invalid_action_never_becomes_a_tool_call(make_model):
    bad_argument = '{"name":"echo","arguments":{"other":"value"}}'
    model, tool = make_model(bad_argument)
    with pytest.raises(ValidationError):
        _generate(model, _runtime_tools(tool))

    bad_number = '{"name":"final_answer","arguments":{"answer":1e999}}'
    model, _ = make_model(bad_number, final_answer_schema={"type": "number"})
    with pytest.raises(ValueError, match="Nonfinite"):
        _generate(model)


def test_tool_schema_changes_fail_before_generation(make_model):
    model, tool = make_model('{"name":"echo","arguments":{"value":"ok"}}')
    tool.inputs["value"]["type"] = "integer"
    with pytest.raises(RuntimeError, match="schema changed"):
        _generate(model, _runtime_tools(tool))


def test_native_agents_use_the_constrained_model(make_model):
    final = '{"name":"final_answer","arguments":{"answer":"varies"}}'
    model, tool = make_model(final)
    tool_agent = ToolCallingAgent(tools=[tool], model=model, max_steps=2)
    assert tool_agent.model is model
    assert tool_agent.run("Answer") == "varies"

    model, tool = make_model(final)
    code_agent = CodeAgent(
        tools=[tool], model=model, max_steps=2,
        instructions="Use the user's requested format.",
    )
    assert code_agent.model is model
    assert code_agent.run("Answer") == "varies"
    system_prompt = model.outlines_model.inputs[0].messages[0]["content"]
    assert "Choose the next single action" in system_prompt
    assert "echo" in system_prompt
    assert "Use the user's requested format." in system_prompt


@pytest.mark.parametrize("agent_cls", [ToolCallingAgent, CodeAgent])
def test_native_agent_rejects_invalid_generated_action(make_model, agent_cls):
    model, tool = make_model('{"name":"echo","arguments":{"other":"bad"}}')
    agent = agent_cls(tools=[tool], model=model, max_steps=2)
    with pytest.raises(AgentGenerationError):
        agent.run("Echo")


def test_native_step_limit_uses_smolagents_fallback(make_model):
    model, tool = make_model('{"name":"echo","arguments":{"value":"again"}}')
    agent = ToolCallingAgent(tools=[tool], model=model, max_steps=1)
    result = agent.run("Repeat")
    assert "Error in generating final LLM output" in str(result)


def test_unsupported_tool_schema_is_rejected(make_model):
    tool = EchoTool()
    tool.inputs = {"value": {"type": "object", "description": "Unsupported"}}
    with pytest.raises(ValueError, match="JSON scalar"):
        # This fails before an Outlines backend is created.
        from smolagents_outlines._core import _action_schema

        _action_schema([tool], None, None)
