"""Example unit test."""

import pytest

failed_msg = "Test failed!"


@pytest.fixture()
def example_fixture_int() -> int:
    """Define example fixture."""
    return 5


def example_function(x: int) -> int:
    """Define example function to test."""
    return x + 1


def example_test(example_fixture_int: int) -> None:
    """Test the example function."""
    if example_function(4) != example_fixture_int:
        raise ValueError(failed_msg)
