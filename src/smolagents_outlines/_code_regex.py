"""Build an Outlines regex for a small, straight-line CodeAgent language.

The generated text is Python, but only 1–8 newline-separated statements are
allowed: ``vN = literal``, ``vN = tool(...)``, ``vN = vM``, a simple numeric
assignment such as ``v2 = v0 + 1``, or a direct declared tool call.
``final_answer(answer=...)`` may be the final statement. Tool calls use every
declared keyword in schema order, with either a ``vN`` reference or a scalar
Python literal admitted by that input's schema. String literals use double
quotes and a deliberately small ASCII alphabet; escapes, multiline strings,
imports, indexing, control flow, and arbitrary Python are outside this subset.

A variable reference has valid call *syntax*, but its runtime value cannot be
checked by a generation-time regex. Most numeric bounds, arbitrary string
patterns, and other constraints that cannot be encoded here remove the literal
branch; the variable branch remains. No wildcard code branch is added.
"""

from __future__ import annotations

import json
import keyword
import math
import re
from collections.abc import Mapping

from jsonschema import Draft202012Validator


_VAR = r"v[0-9]+"
_SAFE_CHARS = r"[A-Za-z0-9 _.,:/!?'-]"
_SAFE_STRING = '"' + _SAFE_CHARS + '*"'
_INTEGER = r"-?(?:0|[1-9][0-9]*)"
_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
_NONNEG_INTEGER = r"(?:0|[1-9][0-9]*)"
_NONNEG_NUMBER = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
_BOOLEAN = r"(?:True|False)"
_SCALAR_LITERAL = rf"(?:{_SAFE_STRING}|{_NUMBER}|{_BOOLEAN}|None)"
_ANNOTATIONS = {"description", "title", "default", "examples"}
_SCALAR_KEYS = {
    "type", "enum", "const", "minLength", "maxLength", "pattern",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
}


def _either(*parts: str | None) -> str | None:
    values = list(dict.fromkeys(part for part in parts if part))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "(?:" + "|".join(values) + ")"


def _safe_string(value: str) -> bool:
    return re.fullmatch(_SAFE_CHARS + "*", value) is not None


def _exact_scalar(value: object) -> str | None:
    if isinstance(value, str):
        if not _safe_string(value):
            return None
        return re.escape(json.dumps(value, ensure_ascii=True))
    if value is None or isinstance(value, bool):
        return re.escape(repr(value))
    if isinstance(value, int):
        return re.escape(str(value))
    if isinstance(value, float) and math.isfinite(value):
        return re.escape(repr(value))
    return None


def _scalar_literal(schema: Mapping[str, object]) -> str | None:
    """Return only literal forms known to satisfy a scalar JSON schema."""
    if "enum" in schema or "const" in schema:
        values = schema.get("enum", [schema.get("const")])
        if not isinstance(values, list):
            return None
        validator = Draft202012Validator(dict(schema))
        return _either(*(
            _exact_scalar(value) for value in values if validator.is_valid(value)
        ))

    if set(schema) - _SCALAR_KEYS - _ANNOTATIONS:
        return None
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else declared
    if not isinstance(types, list):
        return None
    patterns: list[str] = []
    for item in types:
        if item == "string":
            if "pattern" in schema:
                continue
            minimum = schema.get("minLength", 0)
            maximum = schema.get("maxLength")
            if (not isinstance(minimum, int) or isinstance(minimum, bool)
                    or minimum < 0 or (maximum is not None and
                    (not isinstance(maximum, int) or isinstance(maximum, bool)
                     or maximum < minimum))):
                continue
            count = ("*" if minimum == 0 and maximum is None else
                     "{" + str(minimum) + "," +
                     ("" if maximum is None else str(maximum)) + "}")
            patterns.append('"' + _SAFE_CHARS + count + '"')
        elif item in ("integer", "number"):
            if any(key in schema for key in (
                "maximum", "exclusiveMinimum", "exclusiveMaximum"
            )):
                continue
            minimum = schema.get("minimum")
            if minimum is None:
                patterns.append(_INTEGER if item == "integer" else _NUMBER)
            elif (isinstance(minimum, (int, float)) and
                  not isinstance(minimum, bool) and minimum == 0):
                patterns.append(
                    _NONNEG_INTEGER if item == "integer" else _NONNEG_NUMBER
                )
        elif item == "boolean":
            patterns.append(_BOOLEAN)
        elif item == "null":
            patterns.append("None")
    return _either(*patterns)


def _object_literal(schema: Mapping[str, object]) -> str | None:
    """A fixed-key object with scalar fields, each a literal or variable."""
    allowed = {"type", "properties", "required", "additionalProperties"} | _ANNOTATIONS
    if set(schema) - allowed or schema.get("type") != "object":
        return None
    properties = schema.get("properties")
    required = schema.get("required")
    if (not isinstance(properties, Mapping) or not isinstance(required, list)
            or schema.get("additionalProperties") is not False
            or set(required) != set(properties)):
        return None
    fields: list[str] = []
    for key, spec in properties.items():
        if not isinstance(key, str) or not isinstance(spec, Mapping):
            return None
        declared = spec.get("type")
        types = [declared] if isinstance(declared, str) else declared
        if not isinstance(types, list) or not types or any(
            item not in ("string", "integer", "number", "boolean", "null")
            for item in types
        ):
            return None
        fields.append(re.escape(json.dumps(key, ensure_ascii=True)) + ": " + _value(spec))
    return r"\{" + ", ".join(fields) + r"\}"


def _value(schema: Mapping[str, object], *, allow_object: bool = False) -> str:
    return _either(
        _VAR,
        _scalar_literal(schema),
        _object_literal(schema) if allow_object else None,
    ) or _VAR


def _call(name: str, arguments: Mapping[str, object], *, final: bool) -> str:
    properties = arguments.get("properties")
    required = arguments.get("required")
    if (arguments.get("type") != "object" or
            arguments.get("additionalProperties") is not False or
            not isinstance(properties, Mapping) or
            not isinstance(required, list) or
            not set(required).issubset(properties)):
        raise ValueError(f"Unsupported argument object for {name}")
    pieces: list[str] = []
    for arg_name, spec in properties.items():
        if (not isinstance(arg_name, str) or not arg_name.isidentifier() or
                keyword.iskeyword(arg_name) or not isinstance(spec, Mapping)):
            raise ValueError(f"Unsupported argument in {name}")
        pieces.append(
            re.escape(arg_name) + "=" + _value(spec, allow_object=final)
        )
    return re.escape(name) + r"\(" + ", ".join(pieces) + r"\)"


def build_code_regex(action_schema: Mapping[str, object]) -> str:
    """Derive a bounded Python-code regex from ``_action_schema``'s ``oneOf``.

    The returned pattern is intended for ``outlines.types.Regex``. It accepts
    complete programs with one through eight statements; it does not anchor
    variable references to assignments or validate their runtime values.
    """
    branches = action_schema.get("oneOf")
    if not isinstance(branches, list) or not branches:
        raise ValueError("Expected an action schema with oneOf branches")
    custom: list[str] = []
    final_call: str | None = None
    seen: set[str] = set()
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValueError("Invalid action schema branch")
        properties = branch.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("Action branch has no properties")
        name_schema = properties.get("name")
        arguments = properties.get("arguments")
        if not isinstance(name_schema, Mapping) or not isinstance(arguments, Mapping):
            raise ValueError("Action branch has no name or arguments")
        name = name_schema.get("const")
        if (not isinstance(name, str) or not name.isidentifier() or
                keyword.iskeyword(name) or name in seen):
            raise ValueError(f"Invalid or duplicate tool name: {name!r}")
        seen.add(name)
        call = _call(name, arguments, final=name == "final_answer")
        if name == "final_answer":
            final_call = call
        else:
            custom.append(call)
    if not custom or final_call is None:
        raise ValueError("Expected custom tools and one final_answer branch")

    tool_call = _either(*custom)
    assert tool_call is not None
    numeric_atom = _either(_VAR, _NUMBER)
    assert numeric_atom is not None
    arithmetic = numeric_atom + r" [+*/-] " + numeric_atom
    assigned_value = _either(
        tool_call, _SCALAR_LITERAL, _VAR, arithmetic,
    )
    assert assigned_value is not None
    assignment = _VAR + " = " + assigned_value
    statement = _either(assignment, tool_call)
    assert statement is not None
    return "(?:" + statement + r"\n){0,7}(?:" + statement + "|" + final_call + ")"


__all__ = ["build_code_regex"]
