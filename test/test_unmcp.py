from pathlib import Path

import pytest

from agentica.unmcp import MCPFunction


@pytest.mark.asyncio
async def test_unmcp_definition():
    from inspect import isfunction, signature

    import mcp_server

    path = Path(__file__).parent / "mcp_config.json"
    mcp_functions = await MCPFunction.from_json(path)

    real_functions = {
        name: f
        for name, f in mcp_server.__dict__.items()
        if isfunction(f) and getattr(f, "__module__", "") == mcp_server.__name__
    }

    assert set(real_functions.keys()) == {f.__name__ for f in mcp_functions}

    for mcp_function in mcp_functions:
        mcp_function = mcp_function.__wrapped__
        name = mcp_function.__name__
        real_function = real_functions[name]

        real_sig = name + str(signature(real_function))
        # print(real_sig)

        # check signatures
        derived_sig_a = name + str(mcp_function.__signature__)
        derived_sig_b = name + str(signature(mcp_function))
        assert derived_sig_a == real_sig
        assert derived_sig_b == real_sig
        # metadata should match
        assert mcp_function.__doc__ == real_function.__doc__
        assert mcp_function.__name__ == real_function.__name__
        assert mcp_function.__qualname__ == real_function.__qualname__
        assert signature(mcp_function) == signature(real_function)
        assert mcp_function.__annotations__ == real_function.__annotations__

        # check parameter structure
        m_params = list(mcp_function.__signature__.parameters.values())
        r_params = list(signature(real_function).parameters.values())
        assert [p.name for p in m_params] == [p.name for p in r_params]
        for mp, rp in zip(m_params, r_params):
            assert mp.kind == rp.kind
            assert mp.default == rp.default
            assert mp.annotation == rp.annotation

        # check execution using kwargs (assert error for failing tool)
        example = real_function.__example__
        if name == "fail":
            with pytest.raises(Exception) as em:
                await mcp_function(**example)
            with pytest.raises(Exception) as er:
                real_function(**example)
            # We cannot tell the type of exception raised by the tool
            # assert em == er
        else:
            result = await mcp_function(**example)
            assert result == real_function(**example)

        # check execution using positionals (order preserved by schema)
        positional_args = [
            example[k] for k in signature(real_function).parameters.keys() if k in example
        ]
        if name == "fail":
            with pytest.raises(Exception) as em:
                await mcp_function(*positional_args)
            with pytest.raises(Exception) as er:
                real_function(*positional_args)
            # We cannot tell the type of exception raised by the tool
            # assert em == er
        else:
            result_pos = await mcp_function(*positional_args)
            assert result_pos == real_function(*positional_args)
