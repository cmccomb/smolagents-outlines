"""Native-agent cases around the public model adapter."""

import json
from typing import Literal

import pytest
from outlines.types import JsonSchema, Regex
from smolagents import CodeAgent, Tool, ToolCallingAgent, tool
from smolagents.utils import AgentGenerationError

from smolagents_outlines import OutlinesToolModel


box_calls = []


@tool
def describe_box(box: Literal["Compact", "Standard"]) -> str:
    """Describe a box.

    Args:
        box: Box name to describe.
    """
    box_calls.append(box)
    return f"box={box}"


class PairTool(Tool):
    name = "pair"
    description = "Format an integer and a boolean."
    inputs = {
        "count": {"type": "integer", "description": "Integer count"},
        "enabled": {"type": "boolean", "description": "Whether it is enabled"},
    }
    output_type = "string"

    def __init__(self):
        self.calls = []
        super().__init__()

    def forward(self, count: int, enabled: bool) -> str:
        self.calls.append((count, enabled))
        return f"{count}:{enabled}"


class ReadyTool(Tool):
    name = "ready"
    description = "Return a constant."
    inputs = {}
    output_type = "string"

    def __init__(self):
        self.calls = 0
        super().__init__()

    def forward(self) -> str:
        self.calls += 1
        return "ready"


class IntegerTool(Tool):
    name = "integer_value"
    description = "Record an integer."
    inputs = {"value": {"type": "integer", "description": "Integer value"}}
    output_type = "string"

    def __init__(self):
        self.calls = []
        super().__init__()

    def forward(self, value: int) -> str:
        self.calls.append(value)
        return str(value)


class NullableTool(Tool):
    name = "maybe"
    description = "Describe an optional value."
    inputs = {"value": {"type": "string", "description": "Optional text", "nullable": True}}
    output_type = "string"

    def __init__(self):
        self.calls = []
        super().__init__()

    def forward(self, value: str | None) -> str:
        self.calls.append(value)
        return "none" if value is None else value


class FakeOutlines:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.responses)


def _model(monkeypatch, tools, *responses, **kwargs):
    backend = FakeOutlines(*responses)
    monkeypatch.setattr(
        "smolagents_outlines._core.outlines.from_transformers",
        lambda *_: backend,
    )
    return OutlinesToolModel(None, None, tools, **kwargs), backend


@pytest.mark.parametrize("agent_class", [ToolCallingAgent, CodeAgent])
def test_native_agents_execute_exact_multiarg_call_and_final_answer(monkeypatch, agent_class):
    pair = PairTool()
    if agent_class is ToolCallingAgent:
        responses = (
            json.dumps({"name": "pair", "arguments": {"count": 2, "enabled": True}}),
            json.dumps({"name": "final_answer", "arguments": {"answer": "2:True"}}),
        )
        max_steps = 2
    else:
        responses = ('v0 = pair(count=2, enabled=True)\nfinal_answer(answer=v0)',)
        max_steps = 1
    model, backend = _model(monkeypatch, [pair], *responses)
    agent = agent_class(tools=[pair], model=model, max_steps=max_steps)

    assert agent.run("Format the pair") == "2:True"
    assert pair.calls == [(2, True)]
    assert len(backend.calls) == len(responses)


@pytest.mark.parametrize("agent_class", [ToolCallingAgent, CodeAgent])
def test_native_agents_execute_zero_argument_tool(monkeypatch, agent_class):
    ready = ReadyTool()
    if agent_class is ToolCallingAgent:
        responses = (
            '{"name":"ready","arguments":{}}',
            '{"name":"final_answer","arguments":{"answer":"ready"}}',
        )
        max_steps = 2
    else:
        responses = ('v0 = ready()\nfinal_answer(answer=v0)',)
        max_steps = 1
    model, _ = _model(monkeypatch, [ready], *responses)
    agent = agent_class(tools=[ready], model=model, max_steps=max_steps)

    assert agent.run("Check readiness") == "ready"
    assert ready.calls == 1


@pytest.mark.parametrize("agent_class", [ToolCallingAgent, CodeAgent])
def test_native_agents_use_decorated_literal_schema(monkeypatch, agent_class):
    box_calls.clear()
    if agent_class is ToolCallingAgent:
        responses = (
            '{"name":"describe_box","arguments":{"box":"Standard"}}',
            '{"name":"final_answer","arguments":{"answer":"box=Standard"}}',
        )
        max_steps = 2
    else:
        responses = ('v0 = describe_box(box="Standard")\nfinal_answer(answer=v0)',)
        max_steps = 1
    model, _ = _model(monkeypatch, [describe_box], *responses)
    agent = agent_class(tools=[describe_box], model=model, max_steps=max_steps)

    assert describe_box.inputs["box"]["enum"] == ["Compact", "Standard"]
    assert agent.run("Describe Standard") == "box=Standard"
    assert box_calls == ["Standard"]


@pytest.mark.parametrize("agent_class", [ToolCallingAgent, CodeAgent])
def test_native_agents_accept_explicit_nullable_argument(monkeypatch, agent_class):
    maybe = NullableTool()
    if agent_class is ToolCallingAgent:
        responses = (
            '{"name":"maybe","arguments":{"value":null}}',
            '{"name":"final_answer","arguments":{"answer":"none"}}',
        )
        max_steps = 2
    else:
        responses = ('v0 = maybe(value=None)\nfinal_answer(answer=v0)',)
        max_steps = 1
    model, _ = _model(monkeypatch, [maybe], *responses)
    agent = agent_class(tools=[maybe], model=model, max_steps=max_steps)

    assert agent.run("Describe missing value") == "none"
    assert maybe.calls == [None]


def test_same_model_dispatches_to_both_native_agents(monkeypatch):
    ready = ReadyTool()
    model, backend = _model(
        monkeypatch, [ready],
        '{"name":"final_answer","arguments":{"answer":"tool mode"}}',
        'final_answer(answer="code mode")',
    )
    tool_agent = ToolCallingAgent(tools=[ready], model=model, max_steps=1)
    code_agent = CodeAgent(tools=[ready], model=model, max_steps=1)

    assert tool_agent.run("Answer in tool mode") == "tool mode"
    assert code_agent.run("Answer in code mode") == "code mode"
    assert isinstance(backend.calls[0][1]["output_type"], JsonSchema)
    assert isinstance(backend.calls[1][1]["output_type"], Regex)


@pytest.mark.parametrize("agent_class", [ToolCallingAgent, CodeAgent])
def test_native_agents_can_finish_without_custom_tools(monkeypatch, agent_class):
    response = (
        '{"name":"final_answer","arguments":{"answer":"done"}}'
        if agent_class is ToolCallingAgent else 'final_answer(answer="done")'
    )
    model, backend = _model(monkeypatch, [], response)
    agent = agent_class(tools=[], model=model, max_steps=1)

    assert agent.run("Just answer") == "done"
    assert len(backend.calls) == 1


@pytest.mark.parametrize("agent_class", [ToolCallingAgent, CodeAgent])
def test_step_limit_uses_one_constrained_final_generation(monkeypatch, agent_class):
    ready = ReadyTool()
    first = (
        '{"name":"ready","arguments":{}}'
        if agent_class is ToolCallingAgent else 'ready()'
    )
    model, backend = _model(monkeypatch, [ready], first, '"done"')
    agent = agent_class(tools=[ready], model=model, max_steps=1)

    assert agent.run("Check once, then answer") == "done"
    assert ready.calls == 1
    assert len(backend.calls) == 2
    assert isinstance(backend.calls[1][1]["output_type"], JsonSchema)


def test_step_limit_structured_final_fallback(monkeypatch):
    ready = ReadyTool()
    answer = {"box": "Standard", "cost": 6.35}
    schema = {
        "type": "object",
        "properties": {
            "box": {"type": "string", "enum": ["Compact", "Standard"]},
            "cost": {"type": "number", "minimum": 0},
        },
        "required": ["box", "cost"],
        "additionalProperties": False,
    }
    model, backend = _model(
        monkeypatch, [ready],
        '{"name":"ready","arguments":{}}', json.dumps(answer),
        final_answer_schema=schema,
    )
    agent = ToolCallingAgent(tools=[ready], model=model, max_steps=1)

    assert json.loads(agent.run("Check, then report the box")) == answer
    assert ready.calls == 1
    assert len(backend.calls) == 2
    assert isinstance(backend.calls[1][1]["output_type"], JsonSchema)


def test_invalid_final_fallback_fails_closed_without_retry(monkeypatch):
    ready = ReadyTool()
    model, backend = _model(
        monkeypatch, [ready], 'ready()', '"not-an-allowed-answer"',
        final_answers=["done"],
    )
    agent = CodeAgent(tools=[ready], model=model, max_steps=1)

    result = agent.run("Check, then report done")
    assert "Error in generating final LLM output" in str(result)
    assert ready.calls == 1
    assert len(backend.calls) == 2


def test_tool_agent_rejects_unlisted_managed_agent_before_generation(monkeypatch):
    ready = ReadyTool()
    model, backend = _model(monkeypatch, [ready], '{"name":"ready","arguments":{}}')
    delegate = ToolCallingAgent(
        tools=[ready], model=model, max_steps=1,
        name="delegate", description="Handle delegated work.",
    )
    parent = ToolCallingAgent(
        tools=[ready], model=model, managed_agents=[delegate], max_steps=1,
    )

    with pytest.raises(AgentGenerationError, match="tools differ"):
        parent.run("Ask the delegate")
    assert backend.calls == []


def test_code_agent_returns_structured_literal_final_answer(monkeypatch):
    ready = ReadyTool()
    schema = {
        "type": "object",
        "properties": {
            "box": {"type": "string", "enum": ["Compact", "Standard"]},
            "cost": {"type": "number", "minimum": 0},
        },
        "required": ["box", "cost"],
        "additionalProperties": False,
    }
    code = 'final_answer(answer={"box": "Standard", "cost": 6.35})'
    model, _ = _model(monkeypatch, [ready], code, final_answer_schema=schema)

    assert CodeAgent(tools=[ready], model=model, max_steps=1).run("Choose") == {
        "box": "Standard", "cost": 6.35,
    }


@pytest.mark.parametrize(
    "bad_code",
    [
        'pair(2, True)',
        'pair(enabled=True, count=2)',
        'pair(count=2, count=3, enabled=True)',
        'pair(count="two", enabled=True)',
        'pair(count=2, enabled=True, other=1)',
        'missing(count=2, enabled=True)',
    ],
)
def test_code_agent_rejects_bad_tool_call_before_execution(monkeypatch, bad_code):
    pair = PairTool()
    model, backend = _model(monkeypatch, [pair], bad_code)
    agent = CodeAgent(tools=[pair], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Format the pair")
    assert pair.calls == []
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "bad_action",
    [
        '{"name":"pair","arguments":{"count":2}}',
        '{"name":"pair","arguments":{"count":2,"enabled":true,"other":1}}',
        '{"name":"pair","arguments":{"count":"two","enabled":true}}',
        '{"name":"missing","arguments":{"count":2,"enabled":true}}',
    ],
)
def test_tool_agent_rejects_bad_tool_call_before_execution(monkeypatch, bad_action):
    pair = PairTool()
    model, backend = _model(monkeypatch, [pair], bad_action)
    agent = ToolCallingAgent(tools=[pair], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Format the pair")
    assert pair.calls == []
    assert len(backend.calls) == 1


def test_native_agent_rejects_schema_drift_before_generation(monkeypatch):
    pair = PairTool()
    model, backend = _model(monkeypatch, [pair], 'final_answer(answer="unused")')
    pair.inputs = {**pair.inputs, "count": {**pair.inputs["count"], "type": "number"}}
    agent = CodeAgent(tools=[pair], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError, match="schema changed"):
        agent.run("Format the pair")
    assert backend.calls == []


def test_integer_json_action_must_not_pass_python_float_to_tool(monkeypatch):
    """JSON Schema accepts 1.0 as integer, but the Python tool expects int."""
    integer_tool = IntegerTool()
    bad = '{"name":"integer_value","arguments":{"value":1.0}}'
    model, backend = _model(monkeypatch, [integer_tool], bad)
    agent = ToolCallingAgent(tools=[integer_tool], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Record an integer")
    assert integer_tool.calls == []
    assert len(backend.calls) == 1
