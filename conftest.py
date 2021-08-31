# Ignore VCF files during pytest collection, so it doesn't fail if cyvcf2 isn't installed.
collect_ignore_glob = ["sgkit/io/vcf/*.py", ".github/scripts/*.py"]


def pytest_configure(config) -> None:  # type: ignore
    # Add "gpu" marker
    config.addinivalue_line("markers", "gpu:Run tests that run on GPU")


def pytest_sessionfinish(session, exitstatus):  # type: ignore
    from sgkit.utils import func_name_to_input_variable_lists, func_name_to_output_variable_lists

    print("Input variables")
    print()
    for func, var_lists in dict(sorted(func_name_to_input_variable_lists.items())).items():
        print(f"\t{func}")
        for var_list in var_lists:
            print(f"\t\t{var_list}")
    print()

    print("Output variables")
    print()
    for func, var_lists in dict(sorted(func_name_to_output_variable_lists.items())).items():
        print(f"\t{func}")
        for var_list in var_lists:
            print(f"\t\t{var_list}")
    print()

    import sgkit

    missing = set(sgkit.__all__) - set(func_name_to_output_variable_lists.keys())
    print("Missing:")
    for f in sorted(missing):
        print(f)
        

