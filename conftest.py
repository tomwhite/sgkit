# Ignore IO files during pytest collection, so it doesn't fail if cbgen/cyvcf2/plink are not installed.
collect_ignore_glob = ["sgkit/io/*/*.py", ".github/scripts/*.py"]


def pytest_configure(config) -> None:  # type: ignore
    # Add "gpu" marker
    config.addinivalue_line("markers", "gpu:Run tests that run on GPU")
