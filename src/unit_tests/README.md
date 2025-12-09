# Run all tests
python -m src.unit_tests.run_tests

# Verbose mode
python -m src.unit_tests.run_tests -v

# Quiet mode
python -m src.unit_tests.run_tests -q

# Stop on first failure
python -m src.unit_tests.run_tests -f

# Run specific module
python -m src.unit_tests.run_tests -m test_position_manager

# List available modules
python -m src.unit_tests.run_tests --list