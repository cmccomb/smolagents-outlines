"""Contract tests for generated actions and agent integration."""

import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError
from outlines_core.json_schema import build_regex_from_schema
from outlines.types import Regex
from smolagents import CodeAgent, Tool, ToolCallingAgent
from smolagents.models import ChatMessage, MessageRole
from smolagents.utils import AgentGenerationError

from smolagents_outlines import OutlinesToolModel
from smolagents_outlines._code_regex import build_code_regex
from smolagents_outlines._core import _action_schema, _prior_successful_code


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
        self.kwargs = []

    def __call__(self, *args, **kwargs):
        self.inputs.append(args[0])
        self.kwargs.append(kwargs)
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


def _code_model(monkeypatch, code, tools, **kwargs):
    responses = code if isinstance(code, tuple) else (code,)
    fake = FakeOutlines(*responses)
    monkeypatch.setattr(
        "smolagents_outlines._core.outlines.from_transformers",
        lambda *_: fake,
    )
    return OutlinesToolModel(None, None, tools, **kwargs)


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
        'v0 = echo(value="hello")\n'
        "v1 = upper(value=v0)\n"
        "final_answer(answer=v1)"
    )
    model = _code_model(monkeypatch, code, [echo, upper])
    agent = CodeAgent(tools=[echo, upper], model=model, max_steps=1)

    assert agent.run("Uppercase an echo") == "HELLO"
    assert echo.calls == ["hello"]
    assert upper.calls == ["hello"]
    assert len(model.outlines_model.inputs) == 1
    assert isinstance(model.outlines_model.kwargs[0]["output_type"], Regex)


def test_code_agent_uses_variable_from_successful_prior_step(monkeypatch):
    echo, upper = CountingEchoTool(), UpperTool()
    model = _code_model(
        monkeypatch,
        (
            'v0 = echo(value="hello")',
            'v1 = upper(value=v0)\nfinal_answer(answer=v1)',
        ),
        [echo, upper],
    )
    agent = CodeAgent(tools=[echo, upper], model=model, max_steps=2)

    assert agent.run("Echo and uppercase") == "HELLO"
    assert echo.calls == ["hello"]
    assert upper.calls == ["hello"]
    assert len(model.outlines_model.inputs) == 2


def test_code_agent_uses_known_constant_from_successful_prior_step(monkeypatch):
    model = _code_model(
        monkeypatch,
        ('v0 = "A"', 'final_answer(answer=v0)'),
        [], final_answers=["A"],
    )
    agent = CodeAgent(tools=[], model=model, max_steps=2)

    assert agent.run("Return an allowed answer") == "A"
    assert len(model.outlines_model.inputs) == 2


def test_code_agent_does_not_trust_variable_from_failed_prior_step(monkeypatch):
    class FailingTool(Tool):
        name = "fail"
        description = "Raise an error."
        inputs = {}
        output_type = "string"

        def __init__(self):
            self.calls = 0
            super().__init__()

        def forward(self) -> str:
            self.calls += 1
            raise ValueError("tool failed")

    fail = FailingTool()
    model = _code_model(
        monkeypatch,
        ('v0 = "A"\nv1 = fail()', 'final_answer(answer=v0)'),
        [fail], final_answers=["A"],
    )
    agent = CodeAgent(tools=[fail], model=model, max_steps=2)

    with pytest.raises(AgentGenerationError):
        agent.run("Return an allowed answer")
    assert fail.calls == 1
    assert len(model.outlines_model.inputs) == 2


def test_prior_code_reconstruction_skips_observation_followed_by_error():
    messages = [
        ChatMessage(role=MessageRole.USER, content="New task:\nAnswer"),
        ChatMessage(role=MessageRole.ASSISTANT, content='<code>\nv0 = "A"\n</code>'),
        ChatMessage(role=MessageRole.TOOL_RESPONSE, content="Observation:\nA"),
        ChatMessage(role=MessageRole.TOOL_RESPONSE, content="Error:\nFinal check failed"),
    ]
    assert _prior_successful_code(messages) == []


def test_prior_code_reconstruction_starts_at_latest_task():
    messages = [
        ChatMessage(role=MessageRole.USER, content="New task:\nFirst"),
        ChatMessage(role=MessageRole.ASSISTANT, content='<code>\nv0 = "A"\n</code>'),
        ChatMessage(role=MessageRole.TOOL_RESPONSE, content="Observation:\nA"),
        ChatMessage(role=MessageRole.USER, content="New task:\nSecond"),
    ]
    assert _prior_successful_code(messages) == []


def test_code_agent_rejects_invalid_backend_output_before_any_execution(monkeypatch):
    echo = CountingEchoTool()
    code = 'v0 = echo(value="must-not-run")\nimport os\nfinal_answer(answer=v0)'
    model = _code_model(monkeypatch, code, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Answer")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_invalid_syntax_without_retry(monkeypatch):
    echo = CountingEchoTool()
    invalid = 'v0 = echo(value="must-not-run")\nfinal_answer(answer=v0'
    model = _code_model(
        monkeypatch, invalid, [echo],
    )
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Answer")
    assert len(model.outlines_model.inputs) == 1
    assert echo.calls == []


def test_code_agent_rejects_known_invalid_variable_argument_before_execution(monkeypatch):
    echo = CountingEchoTool()
    echo.inputs = deepcopy(echo.inputs)
    echo.inputs["value"]["enum"] = ["A"]
    code = 'v0 = "B"\nv1 = echo(value=v0)\nfinal_answer(answer=v1)'
    model = _code_model(monkeypatch, code, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Use the candidate")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_known_invalid_final_answer_variable(monkeypatch):
    model = _code_model(
        monkeypatch, 'v0 = "B"\nfinal_answer(answer=v0)', [],
        final_answers=["A"],
    )
    agent = CodeAgent(tools=[], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Return an allowed answer")
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_accepts_known_valid_final_answer_variable(monkeypatch):
    model = _code_model(
        monkeypatch, 'v0 = "A"\nfinal_answer(answer=v0)', [],
        final_answers=["A"],
    )
    agent = CodeAgent(tools=[], model=model, max_steps=1)

    assert agent.run("Return an allowed answer") == "A"
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_undefined_variable_before_execution(monkeypatch):
    echo = CountingEchoTool()
    model = _code_model(
        monkeypatch, 'v0 = echo(value=v1)\nfinal_answer(answer=v0)', [echo],
    )
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Echo a value")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_known_wrong_type_before_execution(monkeypatch):
    echo = CountingEchoTool()
    model = _code_model(
        monkeypatch, 'v0 = 2\nv1 = echo(value=v0)\nfinal_answer(answer=v1)', [echo],
    )
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Echo a value")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


@pytest.mark.parametrize("code", [
    'v0 = echo(value="</code>")\nfinal_answer(answer=v0)',
    'v0 = echo(value="final_answer( final_answer=")',
])
def test_code_agent_rejects_native_parser_changes_before_execution(monkeypatch, code):
    echo = CountingEchoTool()
    model = _code_model(monkeypatch, code, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError, match="would change"):
        agent.run("Echo the tag")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_tool_shadowed_by_native_python_function(monkeypatch):
    class CustomLenTool(Tool):
        name = "len"
        description = "Return text length using this custom tool."
        inputs = {"value": {"type": "string", "description": "Input text"}}
        output_type = "string"

        def __init__(self):
            self.calls = []
            super().__init__()

        def forward(self, value: str) -> str:
            self.calls.append(value)
            return str(len(value))

    tool = CustomLenTool()
    model = _code_model(monkeypatch, 'v0 = len(value="abc")', [tool])
    agent = CodeAgent(tools=[tool], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError, match="conflict with native Python functions"):
        agent.run("Use the custom length tool")
    assert tool.calls == []
    assert model.outlines_model.inputs == []


def test_code_agent_rejects_dynamic_value_for_restrictive_schema(monkeypatch):
    echo = CountingEchoTool()
    choose = CountingEchoTool()
    choose.name = "choose"
    choose.inputs = {
        "value": {
            "type": "string", "enum": ["A"], "description": "Allowed choice",
        }
    }
    model = _code_model(
        monkeypatch,
        'v0 = echo(value="A")\nv1 = choose(value=v0)\nfinal_answer(answer=v1)',
        [echo, choose],
    )
    agent = CodeAgent(tools=[echo, choose], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Use the choice")
    assert echo.calls == []
    assert choose.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_dynamic_final_answer_for_enum(monkeypatch):
    echo = CountingEchoTool()
    model = _code_model(
        monkeypatch, 'v0 = echo(value="A")\nfinal_answer(answer=v0)',
        [echo], final_answers=["A"],
    )
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Return an allowed answer")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_accepts_known_valid_enum_variable(monkeypatch):
    echo = CountingEchoTool()
    echo.inputs = deepcopy(echo.inputs)
    echo.inputs["value"]["enum"] = ["A"]
    model = _code_model(
        monkeypatch, 'v0 = "A"\nv1 = echo(value=v0)\nfinal_answer(answer=v1)',
        [echo],
    )
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    assert agent.run("Use the allowed value") == "A"
    assert echo.calls == ["A"]
    assert len(model.outlines_model.inputs) == 1


def test_code_agent_rejects_derived_value_outside_numeric_range(monkeypatch):
    tool = CountingEchoTool()
    tool.inputs = {
        "value": {
            "type": "integer", "minimum": 1, "maximum": 3,
            "description": "Small integer",
        }
    }
    model = _code_model(
        monkeypatch, 'v0 = 2 + 2\nv1 = echo(value=v0)\nfinal_answer(answer=v1)',
        [tool],
    )
    agent = CodeAgent(tools=[tool], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Use a small integer")
    assert tool.calls == []
    assert len(model.outlines_model.inputs) == 1


def test_finite_integer_range_compiles_to_constrained_outlines_schema():
    tool = EchoTool()
    tool.inputs = {
        "value": {
            "type": "integer", "minimum": 1, "maximum": 3,
            "description": "Small integer",
        }
    }
    schema = _action_schema([tool], None, None)
    value_schema = schema["oneOf"][0]["properties"]["arguments"]["properties"]["value"]
    assert value_schema["enum"] == [1, 2, 3]

    pattern = build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(pattern, '{"name":"echo","arguments":{"value":2}}')
    assert not re.fullmatch(pattern, '{"name":"echo","arguments":{"value":4}}')


def test_nullable_integer_range_compiles_to_exact_decoder_choices():
    tool = EchoTool()
    tool.inputs = {
        "value": {
            "type": "integer", "nullable": True,
            "minimum": 1, "maximum": 3,
        }
    }
    schema = _action_schema([tool], None, None)
    value_schema = schema["oneOf"][0]["properties"]["arguments"]["properties"]["value"]
    assert value_schema["enum"] == [1, 2, 3, None]

    pattern = build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(pattern, '{"name":"echo","arguments":{"value":2}}')
    assert re.fullmatch(pattern, '{"name":"echo","arguments":{"value":null}}')
    assert not re.fullmatch(pattern, '{"name":"echo","arguments":{"value":4}}')


def test_finite_integer_choices_cover_nested_final_schema_and_exclusive_bounds():
    final_schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer", "exclusiveMinimum": 1,
                "exclusiveMaximum": 5,
            },
        },
        "required": ["count"],
        "additionalProperties": False,
    }
    schema = _action_schema([], final_schema, None)
    value = schema["oneOf"][-1]["properties"]["arguments"]["properties"]["answer"]
    assert value["properties"]["count"]["enum"] == [2, 3, 4]
    assert "enum" not in final_schema["properties"]["count"]


def test_dead_integer_branch_does_not_reject_valid_final_schema():
    final_schema = {
        "anyOf": [
            {"type": "integer", "minimum": 3, "maximum": 1},
            {"type": "string"},
        ],
    }
    schema = _action_schema([], final_schema, None)
    answer = schema["oneOf"][-1]["properties"]["arguments"]["properties"]["answer"]
    assert answer == final_schema


def test_code_regex_supports_structured_and_free_form_final_answers():
    answer_schema = {
        "type": "object",
        "properties": {
            "selected_box": {"type": "string", "enum": ["Compact", "Standard"]},
            "delivered_price_usd": {"type": "number", "minimum": 0},
        },
        "required": ["selected_box", "delivered_price_usd"],
        "additionalProperties": False,
    }
    pattern = build_code_regex(_action_schema([EchoTool()], answer_schema, None))
    assert re.fullmatch(
        pattern,
        'final_answer(answer={"selected_box": "Standard", "delivered_price_usd": 6.35})',
    )
    assert not re.fullmatch(
        pattern,
        'final_answer(answer={"selected_box": "none", "delivered_price_usd": 6.35})',
    )
    assert not re.fullmatch(
        pattern,
        'final_answer(answer={"selected_box": "Standard", "delivered_price_usd": -6.35})',
    )

    default_pattern = build_code_regex(_action_schema([EchoTool()], None, None))
    assert re.fullmatch(default_pattern, 'final_answer(answer="ok")')
    assert re.fullmatch(default_pattern, 'echo(value="https://example.org/?a=1&b=two#part")')
    assert not re.fullmatch(default_pattern, 'echo(value="bad\\"; bad()")')
    assert re.fullmatch(default_pattern, 'echo(value=v31)')
    assert not re.fullmatch(default_pattern, 'echo(value=v32)')


def test_code_regex_integer_enum_uses_python_int_literals():
    tool = EchoTool()
    tool.inputs = {"value": {"type": "integer", "enum": [1.0, 2]}}
    pattern = build_code_regex(_action_schema([tool], None, None))
    assert re.fullmatch(pattern, "echo(value=1)")
    assert re.fullmatch(pattern, "echo(value=2)")
    assert not re.fullmatch(pattern, "echo(value=1.0)")


def test_tool_enum_choices_are_valid_for_schema_and_native_python_type():
    tool = EchoTool()
    tool.inputs = {"value": {
        "type": "integer", "enum": [1.0, 2, 3],
        "minimum": 2, "maximum": 2,
    }}
    schema = _action_schema([tool], None, None)
    argument = schema["oneOf"][0]["properties"]["arguments"]["properties"]["value"]
    assert argument["enum"] == [2]
    assert tool.inputs["value"]["enum"] == [1.0, 2, 3]

    decoder_regex = build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(decoder_regex, '{"name": "echo", "arguments": {"value": 2}}')
    assert not re.fullmatch(decoder_regex, '{"name": "echo", "arguments": {"value": 1}}')
    assert not re.fullmatch(decoder_regex, '{"name": "echo", "arguments": {"value": 3}}')


def test_integral_float_enum_and_const_become_python_integer_choices():
    tool = EchoTool()
    tool.inputs = {"value": {"type": "integer", "enum": [1.0, 2]}}
    schema = _action_schema([tool], None, None)
    argument = schema["oneOf"][0]["properties"]["arguments"]["properties"]["value"]
    assert argument["enum"] == [1, 2]

    tool.inputs = {"value": {"type": "integer", "const": 1.0}}
    schema = _action_schema([tool], None, None)
    argument = schema["oneOf"][0]["properties"]["arguments"]["properties"]["value"]
    assert argument["const"] == 1


def test_nullable_enum_preserves_none_and_impossible_choices_fail_setup():
    tool = EchoTool()
    tool.inputs = {"value": {"type": "string", "nullable": True, "enum": [None, "A"]}}
    schema = _action_schema([tool], None, None)
    argument = schema["oneOf"][0]["properties"]["arguments"]["properties"]["value"]
    assert argument["enum"] == [None, "A"]

    tool.inputs = {"value": {"type": "integer", "enum": [1], "minimum": 2}}
    with pytest.raises(ValueError, match="no JSON- and Python-valid enum choices"):
        _action_schema([tool], None, None)
    tool.inputs = {"value": {"type": "integer", "const": 1, "minimum": 2}}
    with pytest.raises(ValueError, match="no JSON- and Python-valid const value"):
        _action_schema([tool], None, None)


def test_code_regex_rejects_tool_name_that_variables_can_shadow():
    tool = EchoTool()
    tool.name = "v0"
    with pytest.raises(ValueError, match="conflicts with CodeAgent variables"):
        build_code_regex(_action_schema([tool], None, None))


@pytest.mark.parametrize(
    "invalid",
    [
        'v0 = missing(value="A")\nfinal_answer(answer=v0)',
        'v0 = echo(other="A")\nfinal_answer(answer=v0)',
        'v0 = echo(value="B")\nfinal_answer(answer=v0)',
    ],
)
def test_code_agent_rejects_unknown_tools_keys_and_literal_enum(
    monkeypatch, invalid
):
    echo = CountingEchoTool()
    echo.inputs = deepcopy(echo.inputs)
    echo.inputs["value"]["enum"] = ["A"]
    model = _code_model(monkeypatch, invalid, [echo])
    agent = CodeAgent(tools=[echo], model=model, max_steps=1)

    with pytest.raises(AgentGenerationError):
        agent.run("Answer")
    assert echo.calls == []
    assert len(model.outlines_model.inputs) == 1


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
