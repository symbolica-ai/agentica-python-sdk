import io
import itertools
import sys
import tempfile
from builtins import tuple
from collections.abc import Iterable
from enum import Enum
from types import ModuleType
from typing import Any, NamedTuple, get_args, get_origin
from typing import Any as _Any
from urllib.parse import ParseResult

from agentica_internal.core.anno import *
from agentica_internal.core.utils import cycle_guard
from agentica_internal.warpc.resource.handle import is_virtual_class

from agentica.common import ToolModeStrings

__all__ = [
    'Feature',
    'ComingSoon',
    'JsonModeComingSoon',
    'TryCodeMode',
    'check_value_supported',
    'check_return_type_supported',
    'check_sync_compatible',
]


class Feature(Enum):
    LOCAL_FILE_ACCESS = "Local file access"
    NUMERIC_RANGES = "Numeric range objects"
    BINARY_BUFFERS = "Binary buffer objects"
    URL_OBJECTS = "URL objects"
    NUMERICAL_DATA = "Numerical data"
    MODULES = "Modules"
    TABULAR_DATA = "Tabular data"
    JSON_BASE_EXCEPTIONS = "Richer exceptions in JSON mode"
    BUILTIN_SUBCLASSES = "Subclasses of builtin classes"
    CUSTOM_EXCEPTIONS = "Custom exceptions"
    TYPE = "Type"
    OBJECT = "Object"
    ARBITRARY = "Arbitrary"
    CYCLIC_DATA = "Cyclic data structures"
    NO_JSON_SCHEMA = "Non-JSON compatible"
    RETURN_FUNCTIONS = "Returning functions"
    RETURN_ITERATORS = "Returning iterators"
    SYNC_COMPATIBLE = "Sync agentic functions"
    PYDANTIC_MODELS = "Pydantic support"
    C_EXTENSIONS = "C extension objects"
    CROSS_AGENT = "Passing agent-defined resources to other agents"


class ComingSoon(Exception):
    feature: Feature

    def __init__(self, feature: Feature, mode: ToolModeStrings, detail: str = ""):
        self.feature = feature
        self.mode = mode
        self.detail = detail
        super().__init__()

    def __str__(self) -> str:
        msg = f"{self.feature.value} {self.detail} is Coming Soon to agentica."
        return msg.replace('  ', ' ').replace('s is Coming Soon', 's are Coming Soon')

    def __repr__(self) -> str:
        return f"ComingSoon<{self.feature.name} {self.detail}>"


class JsonModeComingSoon(Exception):
    def __str__(self) -> str:
        return "Json mode is Coming Soon!"

    def __repr__(self) -> str:
        return "JsonModeComingSoon"


class TryCodeMode(Exception):
    def __init__(self, feature: Feature, detail: str = ""):
        self.feature = feature
        self.detail = detail
        super().__init__()

    def __str__(self) -> str:
        msg = f"{self.feature.value} {self.detail} may work in 'code' mode but not in 'json' mode!"
        return msg.replace('  ', ' ')

    def __repr__(self) -> str:
        return f"TryCodeMode<{self.feature.name} {self.detail}>"


def check_sync_compatible(
    globals_scope: dict, arg_types: dict, mode: ToolModeStrings = 'code'
) -> None:
    key = find_sync_problems(globals_scope, arg_types)

    if key:
        detail = f"that can call other functions (namely the provided resource {key!r})"
        raise ComingSoon(Feature.SYNC_COMPATIBLE, mode, detail=detail)


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


def check_value_supported(value: Any, mode: ToolModeStrings = 'code') -> None:
    json = mode == 'json'
    for feature in object_missing_features(value, json):
        raise_feature_error(feature, json)


def check_return_type_supported(return_type: Any, mode: ToolModeStrings = 'code') -> None:
    json = mode == 'json'
    for feature in return_type_missing_features(return_type, json):
        raise_feature_error(feature, json)


def raise_feature_error(feature: Feature, json: bool) -> None:
    if feature is Feature.NO_JSON_SCHEMA:
        raise TryCodeMode(feature)
    raise ComingSoon(feature, 'json' if json else 'code')


# user provided a global object whose type() is cls, e.g. an instance of cls
def object_missing_features(object_: object, json: bool) -> MissingFeatures:
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

            if json:
                yield from json_mode_class_missing_features(obj)

        else:
            if guard(cls):
                return

            if feature := system_class_missing_feature(cls):
                yield feature

            for info in SENDING_CHECKS:
                if info.test_object(obj):
                    yield info.feature

            if is_third_party_c_extension(cls):
                yield Feature.C_EXTENSIONS

            if json:
                yield from json_mode_object_missing_features(obj)

    return scan(object_)


def return_type_missing_features(return_type: Any, json: bool) -> MissingFeatures:
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

            if is_third_party_c_extension(anno):
                yield Feature.C_EXTENSIONS

        elif json:
            yield from json_mode_class_missing_features(anno)

    return scan(return_type)


# shared by both object_class_missing_feature and class_type_missing_feature
def system_class_missing_feature(cls: type) -> Feature | None:
    if cls is type:
        return Feature.TYPE

    if cls is object:
        return Feature.OBJECT

    if cls in BUILTIN_CLASSES:
        return

    if issubclass(cls, bytearray):
        return Feature.BINARY_BUFFERS

    if issubclass(cls, ModuleType):
        return Feature.MODULES

    if issubclass(cls, BUILTIN_CLASSES):
        # this checks for NamedTuple
        if issubclass(cls, tuple) and hasattr(cls, '_fields'):
            return

        # this checks for TypedDict
        if issubclass(cls, dict) and (
            hasattr(cls, '__total__')
            or hasattr(cls, '__required_keys__')
            or hasattr(cls, '__optional_keys__')
        ):
            return

        return Feature.BUILTIN_SUBCLASSES

    return None


def json_mode_object_missing_features(obj: object) -> MissingFeatures:
    if isinstance(obj, type):
        return
    if callable(obj) and has_no_schema_type(obj):
        yield Feature.NO_JSON_SCHEMA


def json_mode_class_missing_features(cls: type) -> MissingFeatures:
    if has_no_schema_type(cls):
        yield Feature.NO_JSON_SCHEMA


def has_no_schema_type(obj: _Any) -> bool:
    return False


def is_stdlib_module(module_name: str | None) -> bool:
    if not module_name or module_name in ('builtins', '__main__'):
        return True

    if hasattr(sys, 'stdlib_module_names'):
        top_level = module_name.split('.')[0]
        if top_level in sys.stdlib_module_names:
            return True

    return False


def is_c_extension_module(module_name: str) -> bool:
    mod = sys.modules.get(module_name)
    if mod is None:
        return False

    mod_file = getattr(mod, '__file__', None)
    if mod_file is None:
        return True

    if mod_file.endswith(('.so', '.pyd', '.dylib')):
        return True

    if not module_name.startswith('_'):
        c_backend_name = f'_{module_name}'
        if c_backend_name in sys.modules:
            backend_mod = sys.modules[c_backend_name]
            backend_file = getattr(backend_mod, '__file__', None)
            if backend_file is None or backend_file.endswith(('.so', '.pyd', '.dylib')):
                return True

    return False


def is_third_party_c_extension(cls: type) -> bool:
    if not isinstance(cls, type):
        return False

    module_name = getattr(cls, '__module__', None)

    if not module_name or module_name in ('builtins', '__main__'):
        return False

    allowed_modules = (
        'io',
        '_io',
        'tempfile',
        'datetime',
        'decimal',
        'fractions',
        'uuid',
        'asyncio',
        'random',
        '_asyncio',
        'concurrent.futures',
        'itertools',
    )

    if module_name.startswith(allowed_modules):
        return False

    if is_c_extension_module(module_name):
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


PATH = FeatureInfo(
    feature=Feature.LOCAL_FILE_ACCESS,
    modules=('os', 'io', 'tempfile', 'typing.IO'),
    classes=(
        io.IOBase,
        tempfile._TemporaryFileWrapper,
        tempfile.TemporaryDirectory,
        tempfile.SpooledTemporaryFile,
    ),
    attrs=(),
)

URL = FeatureInfo(
    feature=Feature.URL_OBJECTS,
    modules=('urllib', 'ipaddress'),
    classes=(ParseResult,),
    attrs=(),
)


NUMPY = FeatureInfo(
    feature=Feature.NUMERICAL_DATA,
    modules=('numpy', 'jax', 'cupy', 'torch', 'tensorflow', 'mxnet', 'xarray', 'dask'),
    classes=(),
    attrs=('ndim', 'to_numpy', '__array__'),
)

TABULAR = FeatureInfo(
    feature=Feature.TABULAR_DATA,
    modules=('pandas', 'polars', 'pyarrow', 'modin', 'dask', 'cudf', 'array'),
    classes=(),
    attrs=('columns', 'dtypes', 'iloc', 'to_numpy'),
)

PYDANTIC = FeatureInfo(
    feature=Feature.PYDANTIC_MODELS,
    modules=('pydantic',),
    classes=(),
    attrs=('model_fields', 'model_validate', 'parse_obj', '__pydantic_core_schema__'),
)

ITERATORS = FeatureInfo(
    feature=Feature.RETURN_ITERATORS,
    modules=('itertools',),
    classes=(zip, map, filter, itertools.count, itertools.cycle, itertools.chain),
    attrs=(),
)

RANGE = FeatureInfo(feature=Feature.NUMERIC_RANGES, modules=(), classes=(range,), attrs=())

SENDING_CHECKS = (URL, NUMPY, TABULAR, PYDANTIC)
RETURNING_CHECKS = SENDING_CHECKS + (
    ITERATORS,
    RANGE,
)

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
