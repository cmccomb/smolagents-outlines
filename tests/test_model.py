"""Contract tests for generated actions and agent integration."""

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


class CountingEchoTool(EchoTool):
    def __init__(self):
        self.calls = []
        super().__init__()

    def forward(self, value: str) -> str:
        self.calls.append(value)
        return value


class UpperTool(Tool):
    name = "upper"
    description = "Uppercase the supplied text."
    inputs = {"value": {"type": "string", "description": "Text to uppercase"}}
    output_type = "string"

    def __init__(self):
        self.calls = []
        super().__init__()

    def forward(self, value: str) -> str:
        self.calls.append(value)
        return value.upper()


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


def _code_model(monkeypatch, code, tools):
    # CodeAgent retries invalid code twice before failing closed.
    response = json.dumps({"code": code})
    fake = FakeOutlines(response, response, response)
    monkeypatch.setattr(
        "smolagents_outlines._core.outlines.from_transformers",
        lambda *_: fake,
    )
    return OutlinesToolModel(None, None, tools)


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
    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string"}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    model, tool = make_model(structured, final_answer_schema=schema)
    agent = ToolCallingAgent(tools=[tool], model=model, max_steps=2)
    assert agent.run("Choose") == {"choice": "Standard"}


def test_invalid_action_never_becomes_a_tool_call(make_model):
    bad_argument = '{"name":"echo","arguments":{"other":"value"}}'
    model, tool = make_model(bad_argument)
    with pytest.raises(ValidationError):
        _generate(model, _runtime_tools(tool))

    bad_number = '{"name":"final_answer","arguments":{"answer":1e999}}'
    model, tool = make_model(bad_number, final_answer_schema={"type": "number"})
    with pytest.raises(ValueError, match="Nonfinite"):
        _generate(model, _runtime_tools(tool))


def test_tool_schema_changes_fail_before_generation(make_model):
    model, tool = make_model('{"name":"echo","arguments":{"value":"ok"}}')
    tool.inputs["value"]["type"] = "integer"
    with pytest.raises(RuntimeError, match="schema changed"):
        _generate(model, _runtime_tools(tool))


def test_native_tool_agent_uses_the_constrained_model(make_model):
    final = '{"name":"final_answer","arguments":{"answer":"varies"}}'
    model, tool = make_model(final)
    tool_agent = ToolCallingAgent(tools=[tool], model=model, max_steps=2)
    assert tool_agent.model is model
    assert tool_agent.run("Answer") == "varies"


def test_native_tool_agent_rejects_invalid_generated_action(make_model):
    model, tool = make_model('{"name":"echo","arguments":{"other":"bad"}}')
    agent = ToolCallingAgent(tools=[tool], model=model, max_steps=2)
    with pytest.raises(AgentGenerationError):
        agent.run("Echo")


def test_code_agent_chains_multiple_tools_with_python_variables(monkeypatch):
    echo, upper = CountingEchoTool(), UpperTool()
    code = (
        'first = echo(value="hello")\n'
        'second = upper(first)\n'
        'final_answer(second)'
    )
    model = _code_model(monkeypatch, code, [echo, upper])
    agent = CodeAgent(tools=[echo, upper], model=model, max_steps=1)

    assert agent.run("Uppercase an echo") == "HELLO"
    assert echo.calls == ["hello"]
    assert upper.calls == ["hello"]


def test_code_agent_rejects_invalid_direct_call_before_any_execution(monkeypatch):
    echo = CountingEchoTool()
    code = 'echo(value="first")\necho(unknown="second")'
    model = _code_model(monkeypatch, code, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError, match="unknown or duplicate keyword"):
        agent.run("Call echo twice")
    assert echo.calls == []


def test_code_agent_regenerates_invalid_python_before_execution(monkeypatch):
    echo = CountingEchoTool()
    invalid = 'echo(value="must-not-run")\nif True print("invalid")'
    valid = 'final_answer("ok")'
    fake = FakeOutlines(
        json.dumps({"code": invalid}),
        json.dumps({"code": valid}),
    )
    monkeypatch.setattr(
        "smolagents_outlines._core.outlines.from_transformers",
        lambda *_: fake,
    )
    model = OutlinesToolModel(None, None, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    assert agent.run("Answer after correcting syntax") == "ok"
    assert len(fake.inputs) == 2
    assert echo.calls == []
    action = agent.memory.steps[1]
    assert action.model_output_message.content == f"<code>\n{valid}\n</code>"


def test_code_agent_accepts_dynamic_tool_argument_syntax(monkeypatch):
    echo = CountingEchoTool()
    echo.inputs = deepcopy(echo.inputs)
    echo.inputs["value"]["enum"] = ["A"]
    code = 'candidate = "B"\nobserved = echo(candidate)\nfinal_answer(observed)'
    model = _code_model(monkeypatch, code, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    assert agent.run("Use the candidate") == "B"
    assert echo.calls == ["B"]


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
