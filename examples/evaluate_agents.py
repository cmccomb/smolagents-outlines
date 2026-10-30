"""Run a bounded live evaluation of both native smolagents agents.

This optional example keeps validity, tool execution, and task completion as
separate observations. It does not retry generations or change agent behavior.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal


CASE_NAMES = ("enum", "integers", "chain", "punctuation", "no_tools")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face Transformers model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES),
        help="Cases to run with each agent (default: all)",
    )
    parser.add_argument(
        "--steps", type=int, default=2,
        help="Maximum native agent steps per run (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Maximum tokens per constrained generation (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="JSON report path (default: a new file in the system temp directory)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use cached model files only; do not fetch from Hugging Face Hub",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.max_new_tokens < 1:
        parser.error("--steps and --max-new-tokens must be positive")
    return args


def default_output() -> Path:
    handle, name = tempfile.mkstemp(prefix="smolagents-outlines-eval-", suffix=".json")
    os.close(handle)
    return Path(name)


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        root = Path(__file__).resolve().parents[1]
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from importlib.metadata import version
    from jsonschema import Draft202012Validator
    from outlines.types import JsonSchema, Regex
    from smolagents import CodeAgent, ToolCallingAgent, tool
    from smolagents.memory import ActionStep
    from smolagents.tools import validate_tool_arguments
    from smolagents.utils import AgentMaxStepsError, AgentParsingError
    from smolagents_outlines import OutlinesToolModel
    from smolagents_outlines._code_regex import build_code_regex
    from smolagents_outlines._code_static import validate_code_action
    from smolagents_outlines._core import _native_code_message, _prior_successful_code
    from transformers import AutoModelForCausalLM, AutoTokenizer

    events: list[dict[str, object]] = []

    @tool
    def choose_color(color: Literal["red", "blue"]) -> str:
        """Return the selected color.

        Args:
            color: Color to select.
        """
        result = f"selected {color}"
        events.append({
            "name": "choose_color", "arguments": {"color": color},
            "observation": result,
        })
        return result

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers.

        Args:
            a: First integer.
            b: Second integer.
        """
        result = a * b
        events.append({
            "name": "multiply", "arguments": {"a": a, "b": b},
            "observation": result,
        })
        return result

    @tool
    def echo(value: str) -> str:
        """Return the supplied text.

        Args:
            value: Text to return.
        """
        events.append({
            "name": "echo", "arguments": {"value": value},
            "observation": value,
        })
        return value

    @tool
    def upper(value: str) -> str:
        """Uppercase the supplied text.

        Args:
            value: Text to uppercase.
        """
        result = value.upper()
        events.append({
            "name": "upper", "arguments": {"value": value},
            "observation": result,
        })
        return result

    cases = {
        "enum": (
            [choose_color], 'Call choose_color(color="red") and report its result.',
            "selected red",
        ),
        "integers": (
            [multiply], "Call multiply(a=3, b=4) and report its result.", "12",
        ),
        "chain": (
            [echo, upper],
            'Call echo(value="hello"), pass its result to upper, and report the uppercase result.',
            "HELLO",
        ),
        "punctuation": (
            [echo], 'Call echo(value="A+B (ok)") and report its exact result.',
            "A+B (ok)",
        ),
        "no_tools": ([], 'Use final_answer to return the string "ready".', "ready"),
    }

    class Capture:
        def __init__(self, backend: object) -> None:
            self.backend = backend
            self.records: list[dict] = []

        def __call__(self, *call_args: object, **call_kwargs: object) -> str:
            record = {
                "constraint": call_kwargs["output_type"],
                "raw": None,
                "raw_is_string": False,
                "backend_error": None,
                "adapter_accepted": None,
                "adapter_error": None,
                "prior_successful_code": [],
            }
            self.records.append(record)
            try:
                raw = self.backend(*call_args, **call_kwargs)
            except Exception as exc:
                record["backend_error"] = f"{type(exc).__name__}: {exc}"
                raise
            record["raw"] = raw if isinstance(raw, str) else repr(raw)
            record["raw_is_string"] = isinstance(raw, str)
            return raw

    def inspect_generated(records: list[dict], model: OutlinesToolModel) -> list[dict]:
        inspected = []
        for record in records:
            raw, constraint = record["raw"], record["constraint"]
            kind = "unknown"
            format_valid = False
            format_error = record["backend_error"]
            python_syntax_valid = None
            static_valid = None
            native_code_identity_valid = None
            if isinstance(constraint, Regex):
                kind = "code_action"
                if record["raw_is_string"]:
                    format_valid = re.fullmatch(constraint.pattern, raw) is not None
                    if not format_valid:
                        format_error = "Generated code does not match its Outlines regex"
                    try:
                        ast.parse(raw)
                        python_syntax_valid = True
                    except SyntaxError:
                        python_syntax_valid = False
                    if python_syntax_valid:
                        try:
                            validate_code_action(
                                raw, model.argument_schemas, model.tool_by_name,
                                prior_successful_code=record["prior_successful_code"],
                            )
                            static_valid = True
                        except Exception:
                            static_valid = False
                        try:
                            _native_code_message(raw)
                            native_code_identity_valid = True
                        except Exception:
                            native_code_identity_valid = False
            elif isinstance(constraint, JsonSchema):
                schema = json.loads(constraint.schema)
                kind = "tool_action" if schema == model.schema else "fallback"
                if record["raw_is_string"]:
                    try:
                        Draft202012Validator(schema).validate(json.loads(raw))
                        format_valid = True
                    except Exception as exc:
                        format_error = f"{type(exc).__name__}: {exc}"
            else:
                format_error = f"Unexpected constraint: {type(constraint).__name__}"
            inspected.append({
                "raw": raw,
                "kind": kind,
                "format_valid": format_valid,
                "format_error": format_error,
                "python_syntax_valid": python_syntax_valid,
                "static_valid": static_valid,
                "native_code_identity_valid": native_code_identity_valid,
                "adapter_accepted": record["adapter_accepted"],
                "adapter_error": record["adapter_error"],
                "prior_successful_code": record["prior_successful_code"],
            })
        return inspected

    def inspect_native_steps(agent: CodeAgent | ToolCallingAgent) -> list[dict]:
        inspected = []
        for step in agent.memory.steps:
            if not isinstance(step, ActionStep):
                continue
            calls = step.tool_calls or []
            native_call_origin = "memory_step" if calls else None
            if not calls and step.model_output_message is not None:
                calls = step.model_output_message.tool_calls or []
                if calls:
                    native_call_origin = "model_output_message"
            native_calls = []
            for call in calls:
                function = getattr(call, "function", None)
                name = function.name if function is not None else call.name
                arguments = function.arguments if function is not None else call.arguments
                native_calls.append({
                    "name": name,
                    "arguments": json.loads(json.dumps(arguments, default=str)),
                })

            code_parse_error = None
            if isinstance(agent, CodeAgent) and step.code_action:
                try:
                    declared_custom_calls = [
                        {"name": node.func.id, "source": ast.unparse(node)}
                        for node in ast.walk(ast.parse(step.code_action))
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in agent.model.tool_by_name
                    ]
                except SyntaxError as exc:
                    declared_custom_calls = []
                    code_parse_error = f"{type(exc).__name__}: {exc}"
            else:
                declared_custom_calls = [
                    call for call in native_calls
                    if call["name"] in agent.model.tool_by_name
                ]
            inspected.append({
                "step_number": step.step_number,
                "native_calls": native_calls,
                "native_call_origin": native_call_origin,
                "declared_custom_calls": declared_custom_calls,
                "code_parse_error": code_parse_error,
                "observation": step.observations,
                "is_final_answer": step.is_final_answer,
                "error": None if step.error is None else {
                    "type": type(step.error).__name__, "message": str(step.error),
                },
                "is_max_steps_fallback": isinstance(step.error, AgentMaxStepsError),
                "native_execution_attempted": (
                    bool(native_calls) and not isinstance(step.error, AgentParsingError)
                ),
                "native_execution_succeeded": (
                    bool(native_calls) and step.error is None
                ),
            })
        return inspected

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=False, local_files_only=args.offline,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, dtype=dtype, token=False,
        local_files_only=args.offline,
    )
    digest = hashlib.sha256()
    for source in (
        inspect.getfile(OutlinesToolModel), inspect.getfile(build_code_regex),
        inspect.getfile(validate_code_action), __file__,
    ):
        path = Path(source)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    output = args.output or default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "model": args.model,
        "package_version": version("smolagents-outlines"),
        "source_sha256": digest.hexdigest(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dependency_versions": {
            package: version(package)
            for package in (
                "smolagents", "outlines", "transformers", "accelerate",
                "jsonschema", "torch",
            )
        },
        "model_revision": getattr(hf_model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "device": device,
        "max_steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "results": [],
    }
    for case_name in args.cases:
        tools, prompt, expected = cases[case_name]
        for agent_class in (ToolCallingAgent, CodeAgent):
            events.clear()
            model = OutlinesToolModel(
                hf_model, tokenizer, tools, max_new_tokens=args.max_new_tokens,
            )
            capture = Capture(model.outlines_model)
            model.outlines_model = capture
            adapter_calls = []
            original_generate = model.generate

            def capture_generate(messages, *call_args, **call_kwargs):
                first_record = len(capture.records)
                prior_code = (
                    _prior_successful_code(messages)
                    if agent_class is CodeAgent else []
                )
                try:
                    message = original_generate(messages, *call_args, **call_kwargs)
                except Exception as exc:
                    adapter_error = f"{type(exc).__name__}: {exc}"
                    accepted = False
                    raise
                else:
                    adapter_error = None
                    accepted = True
                    return message
                finally:
                    for record in capture.records[first_record:]:
                        record["adapter_accepted"] = accepted
                        record["adapter_error"] = adapter_error
                        record["prior_successful_code"] = list(prior_code)
                    adapter_calls.append({
                        "accepted": accepted,
                        "error": adapter_error,
                        "backend_call_count": len(capture.records) - first_record,
                        "prior_successful_code_count": len(prior_code),
                    })

            model.generate = capture_generate
            agent = agent_class(tools=tools, model=model, max_steps=args.steps)
            started = time.monotonic()
            try:
                answer = agent.run(prompt)
                error = None
            except Exception as exc:
                answer = None
                error = f"{type(exc).__name__}: {exc}"
            generated = inspect_generated(capture.records, model)
            native_steps = inspect_native_steps(agent)
            tool_events = []
            for event in events:
                schema = model.argument_schemas[event["name"]]
                try:
                    validate_tool_arguments(
                        model.tool_by_name[event["name"]], event["arguments"]
                    )
                    runtime_native_type_valid = True
                except Exception:
                    runtime_native_type_valid = False
                tool_events.append({
                    **event,
                    "runtime_schema_valid": Draft202012Validator(schema).is_valid(
                        event["arguments"]
                    ),
                    "runtime_native_type_valid": runtime_native_type_valid,
                })
            answer_text = str(answer)
            normal_final_answer = any(step["is_final_answer"] for step in native_steps)
            max_steps_fallback = any(step["is_max_steps_fallback"] for step in native_steps)
            fallback_adapter_succeeded = (
                max_steps_fallback
                and bool(adapter_calls)
                and adapter_calls[-1]["accepted"]
            )
            returned_final_response = (
                error is None
                and answer is not None
                and (normal_final_answer or fallback_adapter_succeeded)
            )
            action_errors = [
                {"step_number": step["step_number"], **step["error"]}
                for step in native_steps
                if step["error"] is not None and not step["is_max_steps_fallback"]
            ]
            completed = returned_final_response and normal_final_answer
            result = {
                "agent": agent_class.__name__, "case": case_name,
                "prompt": prompt, "expected_answer": expected,
                "generated": generated, "adapter_calls": adapter_calls,
                "native_steps": native_steps,
                "tool_events": tool_events,
                "backend_call_count": len(generated),
                "backend_raw_output_count": sum(
                    item["raw"] is not None for item in generated
                ),
                "constrained_format_valid_count": sum(
                    item["format_valid"] for item in generated
                ),
                "adapter_accepted_count_including_fallback": sum(
                    item["accepted"] for item in adapter_calls
                ),
                "action_adapter_accepted_count": sum(
                    item["adapter_accepted"]
                    for item in generated
                    if item["kind"] in ("code_action", "tool_action")
                ),
                "fallback_adapter_accepted_count": sum(
                    item["adapter_accepted"]
                    for item in generated if item["kind"] == "fallback"
                ),
                "native_call_count": sum(len(step["native_calls"]) for step in native_steps),
                "native_execution_attempt_step_count": sum(
                    step["native_execution_attempted"] for step in native_steps
                ),
                "native_execution_success_step_count": sum(
                    step["native_execution_succeeded"] for step in native_steps
                ),
                "declared_custom_call_count": sum(
                    len(step["declared_custom_calls"]) for step in native_steps
                ),
                "executed_custom_call_count": len(tool_events),
                "action_errors": action_errors,
                "answer": answer_text, "error": error,
                "returned_final_response": returned_final_response,
                "normal_final_answer": normal_final_answer,
                "max_steps_fallback": max_steps_fallback,
                "fallback_adapter_succeeded": fallback_adapter_succeeded,
                "completed": completed,
                "answer_correct": returned_final_response and answer_text == expected,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            report["results"].append(result)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(
                f"{case_name:12} {agent_class.__name__:16} "
                f"format={result['constrained_format_valid_count']}/{result['backend_raw_output_count']} "
                f"adapter_all={result['adapter_accepted_count_including_fallback']}/"
                f"{len(adapter_calls)} "
                f"native={result['native_execution_success_step_count']}/"
                f"{result['native_execution_attempt_step_count']} "
                f"tools={len(tool_events)}/{result['declared_custom_call_count']} "
                f"errors={len(action_errors)} final={normal_final_answer} "
                f"fallback={max_steps_fallback} "
                f"correct={result['answer_correct']}",
                flush=True,
            )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
