# smolagents-outlines

Constrain the tool action a smolagents model generates at each agent step with an [Outlines](https://dottxt-ai.github.io/outlines/) JSON schema. The reusable product is a `smolagents.Model` that you pass to an agent.

## Install

Install from the GitHub repository:

```bash
python -m pip install "git+https://github.com/cmccomb/smolagents-outlines.git"
```

For a local checkout, run `python -m pip install -e .`. This release pins the smolagents, Outlines, Transformers, and Accelerate versions used for validation.

## Use the model

Given a Hugging Face Transformers model, its tokenizer, and a list of smolagents tools:

```python
from smolagents_outlines import OutlinesToolModel, OutlinesToolCallingAgent

model = OutlinesToolModel(hf_model, tokenizer, tools)
agent = OutlinesToolCallingAgent(tools=tools, model=model, max_steps=8)
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

The same model works with the CodeAgent helper:

```python
from smolagents_outlines import OutlinesCodeAgent

agent = OutlinesCodeAgent(tools=tools, model=model, max_steps=8)
```

The helpers accept the model explicitly. `OutlinesCodeAgent` supplies a tool-selection prompt and executes one schema-valid tool invocation per step; it does not generate arbitrary Python programs. `OutlinesToolCallingAgent` passes native tool-call objects to smolagents. Both helpers raise on action errors and at the step limit. They reject planning, managed agents, additional base tools, and streamed model output because those paths are outside this action schema.

You can pass the model directly to a native smolagents agent if you prefer its normal error and fallback behavior. Use `OutlinesCodeAgent` for the CodeAgent path when you need the helper's prompt setup and tool-set checks.

## Guarantee and scope

`OutlinesToolModel` builds a JSON schema from the supplied tools' declared `Tool.inputs`, constrains generation with Outlines, and validates the result again before returning an action. It returns a schema-valid action or raises an error. Unsupported input schemas fail during setup. The current implementation supports JSON scalar tool arguments (`string`, `integer`, `number`, `boolean`, and `null`).

Repeated valid calls and any call order are allowed. The package does not judge whether an action is useful or whether a final answer is correct. The tools produce observations in Python; there is no model-generated observation schema or separate execution ledger.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pip install build
python -m build
```

A synthetic example notebook is available in the accompanying [Google Drive folder](https://drive.google.com/drive/folders/1CmRCRrfMGnPk47isT5586qNZZPne7QVt).
