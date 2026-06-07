import run_experiments_tuned


def test_small_memory():
    assert run_experiments_tuned.shared_memory_too_large({"TS": 32}, 2) is False


def test_large_memory():
    assert run_experiments_tuned.shared_memory_too_large({"TS": 128}, 2) is True
