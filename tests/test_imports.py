def test_package_import():
    import trading
    assert hasattr(trading, "__version__")
