# smolagents-outlines

Constrain the tool action a smolagents model generates at each agent step with an [Outlines](https://dottxt-ai.github.io/outlines/) JSON schema. The reusable product is a `smolagents.Model` that you pass to an agent.

## Install

Install from the GitHub repository:

```bash
python -m pip install "git+https://github.com/cmccomb/smolagents-outlines.git"
```

For a local checkout, run `python -m pip install -e .`. The package stays at version 0.1.0 while development continues on GitHub; this version has not been published on PyPI. Its dependencies pin the smolagents, Outlines, Transformers, and Accelerate versions used for validation.

## Use the model

Given a Hugging Face Transformers model, its tokenizer, and a list of smolagents tools:

```python
from smolagents import ToolCallingAgent
from smolagents_outlines import OutlinesToolModel

model = OutlinesToolModel(hf_model, tokenizer, tools)
agent = ToolCallingAgent(tools=tools, model=model, max_steps=8)
answer = agent.run("Use the available tools to answer the question.")
```

The default `final_answer` accepts any JSON string, so the answer can vary with the task. To restrict it to known choices, pass `final_answers`:

```python
model = OutlinesToolModel(
    hf_model, tokenizer, tools,
    final_answers=["Compact", "Standard", "none"],
)
```

For a structured result, pass a JSON Schema as `final_answer_schema` instead:

```python
answer_schema = {
    "type": "object",
    "properties": {
        "selected_box": {"type": "string", "enum": ["Compact", "Standard"]},
        "delivered_price_usd": {"type": "number", "minimum": 0},
    },
    "required": ["selected_box", "delivered_price_usd"],
    "additionalProperties": False,
}
model = OutlinesToolModel(hf_model, tokenizer, tools, final_answer_schema=answer_schema)
```

The agent returns a Python dictionary with those fields. The schema checks its shape and types; the agent still has to derive the values from tool observations. `final_answers` and `final_answer_schema` are mutually exclusive.

The same model works with smolagents' native `CodeAgent`:

```python
from smolagents import CodeAgent

agent = CodeAgent(tools=tools, model=model, max_steps=8)
```

`ToolCallingAgent` receives native tool-call objects. `CodeAgent` receives a complete `<code>` block and can write Python with variables, conditionals, loops, and multiple tool calls in one step. For CodeAgent, Outlines constrains a JSON envelope containing the code string; the model parses that string as Python and checks each direct call to a declared tool for argument names, arity, and duplicate or missing arguments. Literal argument values are also checked against the tool schema. If validation fails, the model tries to regenerate the code, then raises rather than returning invalid code. Give the model and agent the same `tools` list. Leave CodeAgent's default code tags and `use_structured_outputs_internally=False` in place. The model supports action generation; planning and other model-generation paths outside an agent action raise. Native smolagents controls step errors and the step-limit fallback.

## Guarantee and scope

For `ToolCallingAgent`, `OutlinesToolModel` builds a JSON schema from the supplied tools' declared `Tool.inputs`, constrains generation with Outlines, and validates the result again before returning an action. It returns a schema-valid action or raises an error. Unsupported input schemas fail during setup. The current implementation supports JSON scalar tool arguments (`string`, `integer`, `number`, `boolean`, and `null`).

For `CodeAgent`, the guarantee is narrower: returned code is syntactically valid Python, and direct declared-tool calls have valid calling syntax. Literal argument values are schema checked. Python expressions can compute arguments at runtime, so their resulting types and values cannot be guaranteed during generation. The native Python executor handles execution and observations; this package does not add an executor or run ledger.

The package ships a `py.typed` marker. The public constructor accepts a Transformers `PreTrainedModel`, `PreTrainedTokenizerBase`, and a sequence of smolagents `Tool` instances; `generate()` follows the smolagents `Model` signature.

Repeated valid calls and any call order are allowed. The package does not judge whether an action is useful or whether a final answer is correct. The tools produce observations in Python; there is no model-generated observation schema or separate execution ledger.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m mypy src/smolagents_outlines --ignore-missing-imports
python -m pip install build
python -m build
```

See the [five-cell Colab example](examples/structured_custom_tools.ipynb) for `@tool` functions with `Literal` argument choices and a native agent run. The same notebook is available in the accompanying [Google Drive folder](https://drive.google.com/drive/folders/1CmRCRrfMGnPk47isT5586qNZZPne7QVt).
