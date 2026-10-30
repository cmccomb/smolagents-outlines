"""Check what a generated CodeAgent block proves before native execution.

This is a fail-closed generation check, not a Python executor. It propagates
block-local constants and tool output *declarations*. A declaration is useful
for choosing a compatible call shape, but cannot prove a tool's runtime value.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from smolagents import Tool


_UNKNOWN = object()
_ANNOTATIONS = {"description", "title", "default", "examples"}
_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@dataclass(frozen=True)
class _Value:
    constant: object = _UNKNOWN
    declared_type: str | None = None
    fields: Mapping[str, _Value] | None = None


def _python_type_matches(value: object, schema: Mapping[str, object]) -> bool:
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else declared
    if not isinstance(types, list):
        return True  # A valid JSON Schema need not declare a type.
    return any(
        (item == "string" and isinstance(value, str))
        or (item == "integer" and type(value) is int)
        or (item == "number" and type(value) in (int, float))
        or (item == "boolean" and type(value) is bool)
        or (item == "null" and value is None)
        or (item == "object" and isinstance(value, dict))
        or (item == "array" and isinstance(value, list))
        for item in types
    )


def _declared_type_matches(declared: str, schema: Mapping[str, object]) -> bool:
    types = schema.get("type")
    types = [types] if isinstance(types, str) else types
    if not isinstance(types, list):
        return False
    return declared in types or (declared == "integer" and "number" in types)


def _prove(value: _Value, schema: Mapping[str, object], location: str) -> None:
    if value.constant is not _UNKNOWN:
        if not _python_type_matches(value.constant, schema):
            raise ValueError(f"{location} has an incompatible Python type")
        Draft202012Validator(dict(schema)).validate(value.constant)
        return

    if value.fields is not None:
        allowed = {"type", "properties", "required", "additionalProperties"} | _ANNOTATIONS
        properties = schema.get("properties")
        required = schema.get("required")
        if (
            set(schema) - allowed
            or schema.get("type") != "object"
            or not isinstance(properties, Mapping)
            or not isinstance(required, list)
            or not set(required).issubset(value.fields)
            or (schema.get("additionalProperties") is False
                and set(value.fields) - set(properties))
        ):
            raise ValueError(f"{location} cannot be proven to match its object schema")
        for key, field in value.fields.items():
            if key in properties:
                field_schema = properties[key]
                if not isinstance(field_schema, Mapping):
                    raise ValueError(f"{location}.{key} has an unsupported schema")
                _prove(field, field_schema, f"{location}.{key}")
        return

    if value.declared_type is None:
        raise ValueError(f"{location} has no provable type")
    if set(schema) - {"type"} - _ANNOTATIONS or not _declared_type_matches(
        value.declared_type, schema
    ):
        raise ValueError(f"{location} cannot be proven from its declared output type")


def _number(value: _Value, location: str) -> str:
    if value.constant is not _UNKNOWN:
        if type(value.constant) is int:
            return "integer"
        if type(value.constant) is float and math.isfinite(value.constant):
            return "number"
    if value.declared_type in ("integer", "number"):
        return value.declared_type
    raise ValueError(f"{location} is not provably numeric")


def _expression(
    node: ast.expr,
    variables: Mapping[str, _Value],
    arguments: Mapping[str, Mapping[str, object]],
    tools: Mapping[str, Tool],
) -> _Value:
    if isinstance(node, ast.Constant):
        return _Value(constant=node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Variable {node.id} is not assigned in a successful earlier step")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _expression(node.operand, variables, arguments, tools)
        _number(operand, "Unary operand")
        if operand.constant is not _UNKNOWN:
            return _Value(constant=-operand.constant)  # type: ignore[operator]
        return operand
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left = _expression(node.left, variables, arguments, tools)
        right = _expression(node.right, variables, arguments, tools)
        left_type = _number(left, "Left arithmetic operand")
        right_type = _number(right, "Right arithmetic operand")
        if right.constant == 0 and isinstance(node.op, ast.Div):
            raise ValueError("Generated arithmetic divides by zero")
        if left.constant is not _UNKNOWN and right.constant is not _UNKNOWN:
            try:
                result = _ARITHMETIC[type(node.op)](left.constant, right.constant)
            except (ArithmeticError, OverflowError) as exc:
                raise ValueError("Generated arithmetic cannot be evaluated") from exc
            if isinstance(result, float) and not math.isfinite(result):
                raise ValueError("Generated arithmetic is nonfinite")
            return _Value(constant=result)
        result_type = (
            "integer" if left_type == right_type == "integer"
            and not isinstance(node.op, ast.Div) else "number"
        )
        return _Value(declared_type=result_type)
    if isinstance(node, ast.Dict):
        fields: dict[str, _Value] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise ValueError("Generated object needs literal string keys")
            fields[key_node.value] = _expression(value_node, variables, arguments, tools)
        if all(field.constant is not _UNKNOWN for field in fields.values()):
            return _Value(constant={key: field.constant for key, field in fields.items()})
        return _Value(fields=fields)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in arguments or node.args or any(kw.arg is None for kw in node.keywords):
            raise ValueError(f"Generated call {name} has unsupported arguments")
        schema = arguments[name]
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"Generated call {name} has no argument schema")
        if {kw.arg for kw in node.keywords} != set(properties):
            raise ValueError(f"Generated call {name} has the wrong arguments")
        for keyword in node.keywords:
            assert keyword.arg is not None
            field_schema = properties[keyword.arg]
            if not isinstance(field_schema, Mapping):
                raise ValueError(f"Generated call {name}.{keyword.arg} has no schema")
            field = _expression(keyword.value, variables, arguments, tools)
            _prove(field, field_schema, f"{name}.{keyword.arg}")
        if name == "final_answer":
            return _Value(declared_type="any")
        return _Value(declared_type=tools[name].output_type)
    raise ValueError("Generated code contains an unsupported expression")


def _analyze_code(
    code: str,
    variables: dict[str, _Value],
    argument_schemas: Mapping[str, Mapping[str, object]],
    tools: Mapping[str, Tool],
) -> None:
    tree = ast.parse(code)
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("Generated assignment has an unsupported target")
            value = _expression(statement.value, variables, argument_schemas, tools)
            variables[statement.targets[0].id] = value
        elif isinstance(statement, ast.Expr):
            _expression(statement.value, variables, argument_schemas, tools)
        else:
            raise ValueError("Generated code contains an unsupported statement")


def validate_code_action(
    code: str,
    argument_schemas: Mapping[str, Mapping[str, object]],
    tools: Mapping[str, Tool],
    *,
    prior_successful_code: Sequence[str] = (),
) -> None:
    """Check one block using variables from successful blocks in this run.

    Earlier blocks are reconstructed from native agent memory. Analysis never
    executes tool calls or reads the native Python executor's state.
    """
    variables: dict[str, _Value] = {}
    for previous in prior_successful_code:
        _analyze_code(previous, variables, argument_schemas, tools)
    _analyze_code(code, variables, argument_schemas, tools)


__all__ = ["validate_code_action"]
