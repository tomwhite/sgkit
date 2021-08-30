# Ignore VCF files during pytest collection, so it doesn't fail if cyvcf2 isn't installed.
collect_ignore_glob = ["sgkit/io/vcf/*.py", ".github/scripts/*.py"]


def pytest_configure(config) -> None:  # type: ignore
    # Add "gpu" marker
    config.addinivalue_line("markers", "gpu:Run tests that run on GPU")


def pytest_sessionfinish(session, exitstatus):  # type: ignore
    print("Finish!")
    from sgkit.utils import func_name_to_variable_lists

    for f, var_lists in dict(sorted(func_name_to_variable_lists.items())).items():
        print(f)
        for var_list in var_lists:
            print(f"\t{var_list}")
