import collections.abc as A
import io
import tempfile
import typing as T
from dataclasses import dataclass
from urllib.parse import ParseResult, urlparse

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
            check_value_supported(feature)

    def test_url_objects(self):
        features = [
            urlparse('https://www.google.com'),
            ParseResult('https', 'example.com', '/', '', '', ''),
        ]

        for feature in features:
            check_value_supported(feature)


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

    def test_url_objects(self):
        check_return_type_supported(ParseResult)

    def test_custom_dataclasses(self):
        @dataclass
        class MyDataClass:
            value: int

        check_return_type_supported(MyDataClass)

    def test_pydantic_models(self):
        class MyModel(BaseModel):
            value: int
            name: str

        check_return_type_supported(MyModel)

        instance = MyModel(value=42, name="test")
        check_value_supported(instance)

    def test_binary_buffers(self):
        check_return_type_supported(bytearray)

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


class TestExceptionMessages:
    """Tests that exception messages contain the value/type information."""

    def test_check_return_type_supported_includes_callable(self):
        """check_return_type_supported should include Callable in exception for callable types."""
        with pytest.raises(ComingSoon) as exc_info:
            check_return_type_supported(T.Callable[..., int])
        error_str = str(exc_info.value)
        assert "typing.Callable[..., int]" in error_str
