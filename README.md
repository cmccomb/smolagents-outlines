# smolagents-outlines

Constrain the tool action a smolagents model generates at each agent step with an [Outlines](https://dottxt-ai.github.io/outlines/) JSON schema. The reusable product is a `smolagents.Model` that you pass to an agent.

## Install

Install from the GitHub repository:

```bash
python -m pip install "git+https://github.com/cmccomb/smolagents-outlines.git@v0.2.0"
```

For a local checkout, run `python -m pip install -e .`. This release pins the smolagents, Outlines, Transformers, and Accelerate versions used for validation.

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

For another JSON result shape, pass a self-contained JSON Schema as `final_answer_schema` instead. These two options are mutually exclusive.

The same model works with smolagents' native `CodeAgent`:

```python
from smolagents import CodeAgent

agent = CodeAgent(tools=tools, model=model, max_steps=8)
```

`ToolCallingAgent` receives native tool-call objects. `CodeAgent` receives a complete `<code>` block with one declared tool invocation per step, rather than a general Python program. For CodeAgent actions, the model adds a concise tool-selection instruction to the native prompt; agent instructions remain available. Give the model and agent the same `tools` list. Leave CodeAgent's default code tags and `use_structured_outputs_internally=False` in place. The model supports action generation; planning and other model-generation paths outside an agent action raise. Native smolagents controls step errors and the step-limit fallback.

## Guarantee and scope

`OutlinesToolModel` builds a JSON schema from the supplied tools' declared `Tool.inputs`, constrains generation with Outlines, and validates the result again before returning an action. It returns a schema-valid action or raises an error. Unsupported input schemas fail during setup. The current implementation supports JSON scalar tool arguments (`string`, `integer`, `number`, `boolean`, and `null`).

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

See the [synthetic Colab example](examples/structured_custom_tools.ipynb) for a complete tool definition and a model-first agent run. The same notebook is available in the accompanying [Google Drive folder](https://drive.google.com/drive/folders/1CmRCRrfMGnPk47isT5586qNZZPne7QVt).
