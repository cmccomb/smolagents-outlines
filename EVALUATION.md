# Agent evaluation

This project targets native `ToolCallingAgent` and `CodeAgent` in smolagents 1.26.0. The deterministic tests adapt cases from Hugging Face's [agent tests](https://github.com/huggingface/smolagents/blob/v1.26.0/tests/test_agents.py) and [tool tests](https://github.com/huggingface/smolagents/blob/v1.26.0/tests/test_tools.py): decorated `Literal` tools, multiple and zero arguments, nullable inputs, no custom tools, final answers, and the native [step-limit fallback](https://github.com/huggingface/smolagents/blob/v1.26.0/src/smolagents/agents.py#L810-L850). They run both agents with the public `OutlinesToolModel` API and a deterministic backend, then check the action passed to native smolagents and the actual tool calls. Invalid backend outputs must fail before a tool executes, without a retry.

## Deterministic tests

`python -m pytest -q` passes 47 tests, including 29 native-agent matrix cases. The matrix covers:

| Area | Cases |
| --- | --- |
| Native execution | Typed multi-argument tools, zero-argument tools, decorated enum tools, nullable arguments, and Python tool chaining |
| Final answers | Free-form, enumerated, and structured answers; both agents with no custom tools; constrained step-limit fallback |
| Fail-closed behavior | Unknown tools, wrong or extra arguments, malformed code, schema drift, invalid fallback output, and no backend retry |
| Python type edge | A JSON `1.0` satisfies JSON Schema's `integer`, but smolagents rejects it for a Python `int` input; the adapter now rejects the action before returning a native tool call |

## Live model runs

The live probes used cached Qwen2.5 Instruct weights on Apple MPS. Each recorded the raw Outlines output, the constraint supplied to the backend, native tool observations, and whether the agent returned a final response. An action or fallback was counted as format-valid only after checking it against its supplied regex or JSON Schema. These small runs assess behavior; they are not a task-success benchmark.

| Run | Agents and cases | Format-valid outputs | Schema-valid executed tool calls | Final responses | Exact target text |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B before hardening | Both agents: enum, integer, chain | 12/12 | 8/8 | 0/6 | 0/6 |
| Qwen2.5-1.5B before hardening | Both agents: enum | 5/5 | 2/2 | 1/2 | 1/2 |
| Qwen2.5-0.5B after fallback and type fixes | Both agents: enum, integer | 11/11, including 3 fallbacks | 5/5 | 4/4 | 3/4 |
| Qwen2.5-0.5B with no custom tools | Both agents | 4/4 | No custom calls | 2/2 | 2/2 |
| Qwen2.5-0.5B with printable punctuation | Both agents: string tool | 5/5 | 2/2 | 2/2 | 1/2 |
| Qwen2.5-0.5B final full matrix | Both agents: all five cases | 26/26 | 9/9 | 10/10 | 7/10 |

The pre-hardening runs reached `max_steps` without `final_answer`; smolagents then called `model.generate(messages)` for a final fallback, which the adapter previously rejected. The adapter now constrains that one native fallback with the configured final-answer schema. This is an additional final response, not an action retry. In one CodeAgent integer run, the model generated arithmetic with undefined variables, executed no tool, and returned `12` through fallback. A returned answer therefore does not prove tool use or correctness.

“Exact target text” means string equality with the simple expected response, not a judgment of semantic correctness. A separate one-step run using the portable script had both agents return schema-valid final responses, but CodeAgent returned a JSON-looking string instead of the target `ready`.

The final full matrix used the repository's portable `examples/evaluate_agents.py` on the complete hardening changes. CodeAgent answered both the integer and chaining prompts without executing their tools; those answers happened to match the target text. This reinforces why the report records tool execution separately from final responses and exact text matches.

## Contract limits

- `ToolCallingAgent` gets schema-constrained JSON actions and an additional check against smolagents' Python argument types before an action is returned. Native state-variable substitution can still replace a generated string with another runtime value before tool execution.
- `CodeAgent` gets a bounded straight-line Python regex. It fixes direct tool names, exact keyword arguments, and supported literal forms. Variable references can be undefined or hold values outside a tool's schema; neither a regex nor a Python parse check can inspect their runtime values. Variables are limited to `v0`–`v31`, and string literals use printable ASCII without escapes.
- All declared tool arguments are required in generated calls, even when a decorated tool has a Python default. Managed agents, planning, and streaming are outside the current adapter contract.
- Repeated valid calls, weak reasoning, and semantically wrong final answers remain possible. The library adds no executor, observation formatter, call-order rule, or retry loop.

The package stays at version 0.1.0 until a PyPI release.

Reproduce a bounded live run with `python examples/evaluate_agents.py --cases enum integers --steps 2 --output /tmp/smolagents-eval.json`. Add `--offline` to use cached model files only; omit it to download the public model. The JSON report includes raw generated text, the result of the format checks, tool observations, exact target comparison, and source hashes.
