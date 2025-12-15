from agentica.version import __version__


# TODO: test that importlib.metadata.version("symbolica-agentica") == __version__ in package publishing job when we have one
def test_version():
    assert __version__ is not None
    assert __version__ != "0.0.0-dev"
