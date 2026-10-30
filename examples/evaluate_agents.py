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


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from importlib.metadata import version
    from jsonschema import Draft202012Validator
    from outlines.types import JsonSchema, Regex
    from smolagents import CodeAgent, ToolCallingAgent, tool
    from smolagents_outlines import OutlinesToolModel
    from smolagents_outlines._code_regex import build_code_regex
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
            self.records: list[tuple[str, object]] = []

        def __call__(self, *call_args: object, **call_kwargs: object) -> str:
            raw = self.backend(*call_args, **call_kwargs)
            self.records.append((raw, call_kwargs["output_type"]))
            return raw

    def inspect_generated(records: list[tuple[str, object]], model: OutlinesToolModel) -> list[dict]:
        inspected = []
        for raw, constraint in records:
            try:
                if isinstance(constraint, Regex):
                    if re.fullmatch(constraint.pattern, raw) is None:
                        raise ValueError("Generated code does not match its Outlines regex")
                    ast.parse(raw)
                    kind = "action"
                elif isinstance(constraint, JsonSchema):
                    schema = json.loads(constraint.schema)
                    Draft202012Validator(schema).validate(json.loads(raw))
                    kind = "action" if schema == model.schema else "fallback"
                else:
                    raise TypeError(f"Unexpected constraint: {type(constraint).__name__}")
                valid, error = True, None
            except Exception as exc:
                kind = "unknown"
                valid, error = False, f"{type(exc).__name__}: {exc}"
            inspected.append({"raw": raw, "kind": kind, "valid": valid, "error": error})
        return inspected

    def has_final_answer(generated: list[dict], agent_class: type) -> bool:
        for item in generated:
            if item["kind"] == "fallback":
                return True
            if item["kind"] != "action":
                continue
            if agent_class is ToolCallingAgent:
                if json.loads(item["raw"])["name"] == "final_answer":
                    return True
            elif any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "final_answer"
                for node in ast.walk(ast.parse(item["raw"]))
            ):
                return True
        return False

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
    for source in (inspect.getfile(OutlinesToolModel), inspect.getfile(build_code_regex)):
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
            agent = agent_class(tools=tools, model=model, max_steps=args.steps)
            started = time.monotonic()
            try:
                answer = agent.run(prompt)
                error = None
            except Exception as exc:
                answer = None
                error = f"{type(exc).__name__}: {exc}"
            generated = inspect_generated(capture.records, model)
            tool_events = []
            for event in events:
                schema = model.argument_schemas[event["name"]]
                tool_events.append({
                    **event,
                    "runtime_schema_valid": Draft202012Validator(schema).is_valid(
                        event["arguments"]
                    ),
                })
            answer_text = str(answer)
            completed = (
                error is None
                and "Error in generating final LLM output" not in answer_text
                and has_final_answer(generated, agent_class)
            )
            result = {
                "agent": agent_class.__name__, "case": case_name,
                "prompt": prompt, "expected_answer": expected,
                "generated": generated, "tool_events": tool_events,
                "backend_call_count": len(generated),
                "answer": answer_text, "error": error,
                "completed": completed,
                "answer_correct": completed and answer_text == expected,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            report["results"].append(result)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(
                f"{case_name:12} {agent_class.__name__:16} "
                f"valid={sum(item['valid'] for item in generated)}/{len(generated)} "
                f"tools={len(tool_events)} completed={completed} "
                f"correct={result['answer_correct']}",
                flush=True,
            )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
