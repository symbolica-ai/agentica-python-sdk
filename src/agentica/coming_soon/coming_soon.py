import sys
from builtins import tuple
from collections.abc import Iterable
from enum import Enum
from typing import Any, NamedTuple, get_args, get_origin

from agentica_internal.core.anno import *
from agentica_internal.core.utils import cycle_guard
from agentica_internal.warpc.resource.handle import is_virtual_class

__all__ = [
    'Feature',
    'ComingSoon',
    'check_value_supported',
    'check_return_type_supported',
    'check_sync_compatible',
]


class Feature(Enum):
    JSON_BASE_EXCEPTIONS = "Richer exceptions in JSON mode"
    TYPE = "Type"
    OBJECT = "Object"
    ARBITRARY = "Arbitrary"
    CYCLIC_DATA = "Cyclic data structures"
    NO_JSON_SCHEMA = "Non-JSON compatible"
    RETURN_FUNCTIONS = "Returning functions"
    SYNC_COMPATIBLE = "Sync agentic functions"
    CROSS_AGENT = "Passing agent-defined resources to other agents"
    DEEP_LEARNING = "deep learning libraries"


class ComingSoon(Exception):
    feature: Feature

    def __init__(self, feature: Feature, detail: str = ""):
        self.feature = feature
        self.detail = detail
        super().__init__()

    def __str__(self) -> str:
        msg = f"{self.feature.value} {self.detail} is Coming Soon to agentica."
        return msg.replace('  ', ' ').replace('s is Coming Soon', 's are Coming Soon')

    def __repr__(self) -> str:
        return f"ComingSoon<{self.feature.name} {self.detail}>"


def check_sync_compatible(globals_scope: dict, arg_types: dict) -> None:
    key = find_sync_problems(globals_scope, arg_types)

    if key:
        detail = f"that can call other functions (namely the provided resource {key!r})"
        raise ComingSoon(Feature.SYNC_COMPATIBLE, detail=detail)


def find_sync_problems(globals_scope: dict, arg_types: dict) -> str | None:
    for key, val in globals_scope.items():
        if key.startswith('__'):
            continue
        if callable(val):
            return key

    for key, typ in arg_types.items():
        if key.startswith('__'):
            continue
        if key == 'return':
            continue
        if is_callable_anno(typ):
            return key

    return None


type MissingFeatures = Iterable[Feature]


def _qualified_type_name(t: type | Any) -> str:
    """Return a fully qualified type name like 'numpy.ndarray' or 'builtins.int'."""
    module = getattr(t, '__module__', None)
    qualname = getattr(t, '__qualname__', None) or getattr(t, '__name__', repr(t))
    if module and module != 'builtins':
        return f"{module}.{qualname}"
    return qualname


def check_value_supported(value: Any) -> None:
    for feature in object_missing_features(value):
        type_name = _qualified_type_name(type(value))
        raise_feature_error(feature, f"(namely the provided value of type {type_name}: {value!r})")


def check_return_type_supported(return_type: Any) -> None:
    for feature in return_type_missing_features(return_type):
        type_name = (
            _qualified_type_name(return_type)
            if isinstance(return_type, type)
            else repr(return_type)
        )
        raise_feature_error(feature, f"(namely the specified return type: {type_name})")


def raise_feature_error(feature: Feature, detail: str = "") -> None:
    raise ComingSoon(feature, detail)


# user provided a global object whose type() is cls, e.g. an instance of cls
def object_missing_features(object_: object) -> MissingFeatures:
    guard = cycle_guard()

    def scan(obj):
        cls = type(obj)

        if cls in ATOMIC_CLASSES:
            pass

        elif cls in CONTAINER_CLASSES:
            if len(obj) == 0:
                return

            elif guard(obj):
                yield Feature.CYCLIC_DATA
                return

            elif cls is dict:
                for k, v in obj.items():
                    yield from scan(k)
                    yield from scan(v)
            else:
                for v in obj:
                    yield from scan(v)

        elif is_virtual_class(cls):
            yield Feature.CROSS_AGENT

        elif cls is type or isinstance(obj, type):
            if guard(obj):
                return

            if is_virtual_class(obj):
                yield Feature.CROSS_AGENT

            if feature := system_class_missing_feature(obj):
                yield feature

            for info in SENDING_CHECKS:
                if info.test_class(obj):
                    yield info.feature

        else:
            if guard(cls):
                return

            if feature := system_class_missing_feature(cls):
                yield feature

            for info in SENDING_CHECKS:
                if info.test_object(obj):
                    yield info.feature

    return scan(object_)


def return_type_missing_features(return_type: Any) -> MissingFeatures:
    guard = cycle_guard()

    def scan(anno):
        if anno is Any:
            yield Feature.ARBITRARY

        elif anno is ... or anno is None:
            return

        cls = type(anno)

        if guard(anno):
            return

        if is_callable_anno(anno):
            yield Feature.RETURN_FUNCTIONS

        elif is_anno_class(cls):
            try:
                orig = get_origin(anno)
                if orig is not None:
                    yield from scan(orig)
            except:
                pass

            try:
                for arg in get_args(anno):
                    yield from scan(arg)
            except:
                pass

        elif cls is type or isinstance(anno, type):
            if feature := system_class_missing_feature(anno):
                yield feature

            for info in RETURNING_CHECKS:
                if info.test_class(anno):
                    yield info.feature

    return scan(return_type)


# shared by both object_class_missing_feature and class_type_missing_feature
def system_class_missing_feature(cls: type) -> Feature | None:
    if cls is type:
        return Feature.TYPE

    if cls is object:
        return Feature.OBJECT

    if cls in BUILTIN_CLASSES:
        return

    return None


def is_stdlib_module(module_name: str | None) -> bool:
    if not module_name or module_name in ('builtins', '__main__'):
        return True

    if hasattr(sys, 'stdlib_module_names'):
        top_level = module_name.split('.')[0]
        if top_level in sys.stdlib_module_names:
            return True

    return False


type types = tuple[type, ...]
type strings = tuple[str, ...]


class FeatureInfo(NamedTuple):
    feature: Feature
    modules: strings
    classes: types
    attrs: strings

    def test_class(self, cls: type):
        if not isinstance(cls, type):
            return False
        if cls in self.classes:
            return True
        if issubclass(cls, self.classes):
            return True
        module = getattr(cls, '__module__', None)
        if self.test_module(module):
            return True
        for attr in self.attrs:
            if hasattr(cls, attr):
                return True
        return False

    def test_object(self, obj: Any):
        cls = type(obj)
        if self.test_class(cls):
            return True
        for attr in self.attrs:
            if hasattr(obj, attr):
                return True
        return False

    def test_module(self, module: str | None):
        if type(module) is not str:
            return False
        return module.startswith(self.modules)


DEEP_LEARNING = FeatureInfo(
    feature=Feature.DEEP_LEARNING,
    modules=('jax', 'cupy', 'torch', 'tensorflow'),
    classes=(),
    attrs=(),
)

SENDING_CHECKS = RETURNING_CHECKS = (DEEP_LEARNING,)

ATOMIC_CLASSES: types = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
)

CONTAINER_CLASSES: types = (
    list,
    tuple,
    dict,
    set,
    frozenset,
)

BUILTIN_CLASSES = ATOMIC_CLASSES + CONTAINER_CLASSES
