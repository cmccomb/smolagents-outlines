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

`ToolCallingAgent` receives native tool-call objects. `CodeAgent` receives a complete `<code>` block. Its Outlines regex permits a small straight-line Python subset: up to eight assignments or direct tool calls, with intermediate variables named `v0`, `v1`, and so on. Each declared tool call uses its exact keyword arguments; each argument is a `v`-numbered variable or a literal shaped like the declared scalar type. Enum literals are limited to the declared choices. Generation happens once. A final full-match and Python parse check fail closed if generation is truncated or a backend violates the pattern; there is no retry or repair. Give the model and agent the same `tools` list. Leave CodeAgent's default code tags and `use_structured_outputs_internally=False` in place. The model supports action generation; planning and other model-generation paths outside an agent action raise. Native smolagents controls step errors and the step-limit fallback.

## Guarantee and scope

For `ToolCallingAgent`, `OutlinesToolModel` builds a JSON schema from the supplied tools' declared `Tool.inputs`, constrains generation with Outlines, and validates the result again before returning an action. It returns a schema-valid action or raises an error. Unsupported input schemas fail during setup. The current implementation supports JSON scalar tool arguments (`string`, `integer`, `number`, `boolean`, and `null`).

For `CodeAgent`, generation is constrained to that Python subset. The regex fixes direct tool names, argument names and count, and literal shapes. It does not prove that a variable is defined or that its runtime value matches the argument schema. Numeric ranges and arbitrary string patterns also need runtime checking if they matter. The native Python executor handles execution and observations; this package does not add an executor or run ledger. A structured final-answer schema is guaranteed by Outlines in `ToolCallingAgent`; a variable passed to `CodeAgent`'s `final_answer` has no generation-time value guarantee.

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
