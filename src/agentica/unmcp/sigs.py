import inspect
from typing import Any


def build_signature_and_annotations(
    input_schema: dict[str, Any] | None,
    output_schema: dict[str, Any] | None,
):
    parameters: list[inspect.Parameter] = list()
    annotations: dict[str, Any] = dict()

    in_schema: dict[str, Any] = input_schema
    obj_schema = _first_object_schema(in_schema)
    properties: dict[str, Any] = obj_schema.get("properties", dict()) if obj_schema else dict()
    required: set[str] = set(obj_schema.get("required", list())) if obj_schema else set()

    for name, prop in properties.items():
        t = _schema_to_type(prop)
        annotations[name] = t
        has_default = (name not in required) or ("default" in prop)
        default_val = prop.get("default", None) if has_default else inspect._empty
        parameters.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default_val,
                annotation=t,
            )
        )

    # **kwargs only if explicitly allowed
    additional = obj_schema.get("additionalProperties") if obj_schema else None
    if additional is True or isinstance(additional, dict):
        parameters.append(inspect.Parameter(name="kwargs", kind=inspect.Parameter.VAR_KEYWORD))

    return_t = _schema_to_return_type(output_schema)
    annotations["return"] = return_t

    signature = inspect.Signature(parameters=parameters, return_annotation=return_t)

    return signature, annotations


def _schema_scalar_type(schema: dict[str, Any]) -> Any:
    typ = schema.get("type")
    if typ == "string":
        return str
    if typ == "integer":
        return int
    if typ == "number":
        return float
    if typ == "boolean":
        return bool
    if typ == "null":
        return type(None)
    if typ == "array":
        # tuples via prefixItems
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list) and prefix_items:
            elem_types = [_schema_to_type(s) if isinstance(s, dict) else Any for s in prefix_items]
            try:
                return tuple.__class_getitem__(tuple(elem_types))
            except Exception:
                return tuple
        # homogeneous lists via items
        item_schema = schema.get("items")
        item_t = _schema_to_type(item_schema) if isinstance(item_schema, dict) else Any
        try:
            return list[item_t]
        except Exception:
            return list
    if typ == "object":
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            value_t = _schema_to_type(additional)
            try:
                return dict[str, value_t]
            except Exception:
                return dict
        try:
            return dict[str, Any]
        except Exception:
            return dict
    return Any


def _schema_to_type(schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return Any

    lit = _schema_enum_or_const_type(schema)
    if lit is not None:
        return lit

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            return _schema_union_type([v for v in variants if isinstance(v, dict)])

    typ = schema.get("type")
    if isinstance(typ, list):
        variants = [{"type": t} if not isinstance(t, dict) else t for t in typ]
        return _schema_union_type(variants)

    if isinstance(schema.get("properties"), dict):
        additional = schema.get("additionalProperties")
        return _schema_scalar_type({"type": "object", "additionalProperties": additional})

    return _schema_scalar_type(schema)


def _schema_enum_or_const_type(schema: dict[str, Any]) -> Any | None:
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        from typing import Literal as _Literal

        return _Literal[tuple(schema["enum"])]
    if "const" in schema:
        from typing import Literal as _Literal

        return _Literal[schema["const"]]
    return None


def _schema_union_type(variants: list[dict[str, Any]]) -> Any:
    union_t: Any | None = None
    for v in variants:
        vt = _schema_to_type(v)
        union_t = vt if union_t is None else (union_t | vt)
    return union_t if union_t is not None else Any


def _first_object_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(schema.get("properties"), dict):
        return schema
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, dict) and isinstance(v.get("properties"), dict):
                    return v
    return None


def _schema_to_return_type(output_schema: dict[str, Any] | None) -> Any:
    if not isinstance(output_schema, dict):
        return Any

    # unwrap FastMCP object-with-result if present
    if isinstance(output_schema.get("properties"), dict):
        props: dict[str, Any] = output_schema.get("properties", {})
        if set(props.keys()) == {"result"} and isinstance(props["result"], dict):
            result_schema = props["result"]
            # special-case null -> None for return annotation
            if result_schema.get("type") == "null" or result_schema.get("const", object()) is None:
                return None
            return _schema_to_type(result_schema)

    # top-level enum/const/anyOf/oneOf
    lit = _schema_enum_or_const_type(output_schema)
    if lit is not None:
        # If the literal is None, represent as None not NoneType
        try:
            from typing import get_args as _get_args

            vals = _get_args(lit)
            if len(vals) == 1 and vals[0] is None:
                return None
        except Exception:
            pass
        return lit
    for key in ("anyOf", "oneOf"):
        variants = output_schema.get(key)
        if isinstance(variants, list) and variants:
            return _schema_union_type([v for v in variants if isinstance(v, dict)])

    # objects and the rest
    if output_schema.get("type") == "null":
        return None
    return _schema_to_type(output_schema)
