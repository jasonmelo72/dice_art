import pytest

# Basic fixture example that can be used across test files
@pytest.fixture(scope="session")
def base_setup():
    # Setup code that runs before tests
    print("\nSetting up test environment")
    yield
    # Teardown code that runs after tests
    print("\nTearing down test environment")

@pytest.fixture(scope="function")
def clean_environment():
    # Setup before each test function
    yield
    # Cleanup after each test function