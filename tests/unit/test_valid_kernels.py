import run_experiments_tuned


def test_k5_valid_config():
    assert run_experiments_tuned.valid_kernel5({"TS": 32, "WPT": 8, "TSDK": 16}) is True


def test_k5_ts_not_divisible_by_wpt():
    assert run_experiments_tuned.valid_kernel5({"TS": 32, "WPT": 7, "TSDK": 16}) is False


def test_k5_tsdk_greater_than_ts():
    assert run_experiments_tuned.valid_kernel5({"TS": 16, "WPT": 8, "TSDK": 32}) is False


def test_k5_ts_not_divisible_by_tsdk():
    assert run_experiments_tuned.valid_kernel5({"TS": 32, "WPT": 8, "TSDK": 12}) is False


def test_k6_valid_config():
    config = {"TSM": 128, "TSN": 128, "WPTM": 8, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel6(config) is True


def test_k6_tsm_not_divisible_by_wptm():
    config = {"TSM": 128, "TSN": 128, "WPTM": 7, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k6_tsn_not_divisible_by_wptn():
    config = {"TSM": 128, "TSN": 128, "WPTM": 8, "WPTN": 7}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k6_rtsm_zero():
    config = {"TSM": 64, "TSN": 128, "WPTM": 128, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k6_rtsn_zero():
    config = {"TSM": 128, "TSN": 64, "WPTM": 8, "WPTN": 128}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k7_to_10_valid_config():
    config = {"WIDTH": 4, "TSM": 128, "TSN": 128, "WPTM": 8, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is True


def test_k7_to_10_tsm_not_divisible_by_wptm():
    config = {"WIDTH": 4, "TSM": 128, "TSN": 128, "WPTM": 7, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is False


def test_k7_to_10_tsm_not_divisible_by_width():
    config = {"WIDTH": 8, "TSM": 128, "TSN": 128, "WPTM": 8, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is True
    config["TSM"] = 100
    assert run_experiments_tuned.valid_kernel7_to_10(config) is False


def test_k7_to_10_tsn_not_divisible_by_width():
    config = {"WIDTH": 4, "TSM": 128, "TSN": 100, "WPTM": 8, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is False


def test_k7_to_10_width_1_is_valid():
    config = {"WIDTH": 1, "TSM": 128, "TSN": 128, "WPTM": 8, "WPTN": 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is True
