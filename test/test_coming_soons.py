import collections.abc as A
import io
import itertools
import sqlite3
import tempfile
import typing as T
from dataclasses import dataclass
from types import ModuleType
from urllib.parse import ParseResult, urlparse

import numpy as np
import pandas as pd
import pytest
from agentica_internal.warpc.resource.handle import ResourceHandle
from pydantic import BaseModel

from agentica.coming_soon import *


class TestValidateFeature:
    def test_sync_compatible(self):
        class MyClass:
            pass

        def my_func():
            pass

        check_sync_compatible({'global': 5}, {'local': str})

        with pytest.raises(ComingSoon):
            check_sync_compatible({'global': MyClass}, {})

        with pytest.raises(ComingSoon):
            check_sync_compatible({'global': my_func}, {})

        with pytest.raises(ComingSoon):
            check_sync_compatible({}, {'local': T.Callable})

        with pytest.raises(ComingSoon):
            check_sync_compatible({}, {'local': A.Callable})

        with pytest.raises(ComingSoon):
            check_sync_compatible({}, {'local': T.Callable[..., int]})

    def test_local_file_access(self):
        features = [
            open(__file__, 'r'),  # Use current file instead of /etc/passwd
            io.StringIO('test'),
            io.BytesIO(b'test'),
            tempfile.NamedTemporaryFile(),
        ]

        for feature in features:
            check_value_supported(feature)

    def test_binary_buffers(self):
        features = [
            bytearray(10),
            bytearray(b'hello'),
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_value_supported(feature)

    def test_modules(self):
        features = [
            ModuleType('my_module'),
            itertools,  # actual module
            io,  # actual module
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_value_supported(feature)

    def test_numerical_data(self):
        features = [
            np.array([1, 2, 3]),
            np.zeros((3, 3)),
            np.ones(5),
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_value_supported(feature)

    def test_tabular_data(self):
        features = [
            pd.DataFrame({'a': [1, 2, 3]}),
            pd.Series([1, 2, 3]),
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_value_supported(feature)

    def test_url_objects(self):
        features = [
            urlparse('https://www.google.com'),
            ParseResult('https', 'example.com', '/', '', '', ''),
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_value_supported(feature)

    def test_c_extensions_detected(self):
        with sqlite3.connect(':memory:') as conn:
            with pytest.raises(ComingSoon) as exc_info:
                check_value_supported(conn)
            assert exc_info.value.feature == Feature.C_EXTENSIONS

        bio = io.BytesIO(b'test')
        check_value_supported(bio)

        with open(__file__, 'r') as f:
            check_value_supported(f)


class TestValidateReturn:
    # Should raise

    def test_type(self):
        with pytest.raises(ComingSoon):
            check_return_type_supported(type)

    def test_returns_function(self):
        with pytest.raises(ComingSoon):
            check_return_type_supported(T.Callable)

        with pytest.raises(ComingSoon):
            check_return_type_supported(A.Callable)

        with pytest.raises(ComingSoon):
            check_return_type_supported(T.Callable[..., int])

        with pytest.raises(ComingSoon):
            check_return_type_supported(A.Callable[..., int])

    def test_builtin_subclasses(self):
        check_return_type_supported(int)
        check_return_type_supported(list)
        check_return_type_supported(bool)
        check_return_type_supported(str)
        check_return_type_supported(dict)

        class int2(int): ...

        class list2(int): ...

        class bool2(int): ...

        class str2(str): ...

        class dict2(dict): ...

        with pytest.raises(ComingSoon):
            check_return_type_supported(int2)

        with pytest.raises(ComingSoon):
            check_return_type_supported(list2)

        with pytest.raises(ComingSoon):
            check_return_type_supported(bool2)

        with pytest.raises(ComingSoon):
            check_return_type_supported(str2)

        with pytest.raises(ComingSoon):
            check_return_type_supported(dict2)

    def test_advanced_iterators(self):
        features = [
            itertools.count,
            itertools.cycle,
            itertools.chain,
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_return_type_supported(feature)
            with pytest.raises(ComingSoon):
                check_return_type_supported(feature, mode='json')

    def test_ranges(self):
        with pytest.raises(ComingSoon):
            check_return_type_supported(range)
        with pytest.raises(ComingSoon):
            check_return_type_supported(range, mode='json')

    def test_modules(self):
        with pytest.raises(ComingSoon):
            check_return_type_supported(ModuleType)
        with pytest.raises(ComingSoon):
            check_return_type_supported(ModuleType, mode='json')

    def test_numerical_data(self):
        with pytest.raises(ComingSoon):
            check_return_type_supported(np.ndarray)
        with pytest.raises(ComingSoon):
            check_return_type_supported(np.ndarray, mode='json')

    def test_tabular_data(self):
        features = [
            pd.DataFrame,
            pd.Series,
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_return_type_supported(feature)
            with pytest.raises(ComingSoon):
                check_return_type_supported(feature, mode='json')

    def test_url_objects(self):
        with pytest.raises(ComingSoon):
            check_return_type_supported(ParseResult)
        with pytest.raises(ComingSoon):
            check_return_type_supported(ParseResult, mode='json')

    def test_custom_dataclasses(self):
        @dataclass
        class MyDataClass:
            value: int

        check_return_type_supported(MyDataClass, mode='code')
        check_return_type_supported(MyDataClass, mode='json')

    def test_pydantic_models(self):
        class MyModel(BaseModel):
            value: int
            name: str

        with pytest.raises(ComingSoon) as exc_info:
            check_return_type_supported(MyModel, mode='code')
        assert exc_info.value.feature == Feature.PYDANTIC_MODELS

        with pytest.raises(ComingSoon) as exc_info:
            check_return_type_supported(MyModel, mode='json')
        assert exc_info.value.feature == Feature.PYDANTIC_MODELS

        instance = MyModel(value=42, name="test")
        with pytest.raises(ComingSoon) as exc_info:
            check_value_supported(instance, mode='code')
        assert exc_info.value.feature == Feature.PYDANTIC_MODELS

        with pytest.raises(ComingSoon) as exc_info:
            check_value_supported(instance, mode='json')
        assert exc_info.value.feature == Feature.PYDANTIC_MODELS

    def test_binary_buffers(self):
        features = [
            bytearray,
        ]

        for feature in features:
            with pytest.raises(ComingSoon):
                check_return_type_supported(feature)
            with pytest.raises(ComingSoon):
                check_return_type_supported(feature, mode='json')

    def test_c_extensions_detected_return(self):
        with pytest.raises(ComingSoon) as exc_info:
            check_return_type_supported(sqlite3.Connection)
        assert exc_info.value.feature == Feature.C_EXTENSIONS

    def test_cross_agent(self):
        class VirtualClass:
            ___vhdl___ = ResourceHandle()

        virtual_object = VirtualClass()
        with pytest.raises(ComingSoon) as exc_info:
            check_value_supported(virtual_object)
        assert exc_info.value.feature == Feature.CROSS_AGENT
        with pytest.raises(ComingSoon) as exc_info:
            check_value_supported(VirtualClass)
        assert exc_info.value.feature == Feature.CROSS_AGENT
