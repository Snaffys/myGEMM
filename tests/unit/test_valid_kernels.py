import run_experiments_tuned
from shared import PARAM_TS, PARAM_TSDK, PARAM_TSM, PARAM_TSN, PARAM_WIDTH, PARAM_WPT, PARAM_WPTM, PARAM_WPTN


def test_k5_valid_config():
    assert run_experiments_tuned.valid_kernel5({PARAM_TS: 32, PARAM_WPT: 8, PARAM_TSDK: 16}) is True


def test_k5_ts_not_divisible_by_wpt():
    assert run_experiments_tuned.valid_kernel5({PARAM_TS: 32, PARAM_WPT: 7, PARAM_TSDK: 16}) is False


def test_k5_tsdk_greater_than_ts():
    assert run_experiments_tuned.valid_kernel5({PARAM_TS: 16, PARAM_WPT: 8, PARAM_TSDK: 32}) is False


def test_k5_ts_not_divisible_by_tsdk():
    assert run_experiments_tuned.valid_kernel5({PARAM_TS: 32, PARAM_WPT: 8, PARAM_TSDK: 12}) is False


def test_k6_valid_config():
    config = {PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 8, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel6(config) is True


def test_k6_tsm_not_divisible_by_wptm():
    config = {PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 7, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k6_tsn_not_divisible_by_wptn():
    config = {PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 8, PARAM_WPTN: 7}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k6_rtsm_zero():
    config = {PARAM_TSM: 64, PARAM_TSN: 128, PARAM_WPTM: 128, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k6_rtsn_zero():
    config = {PARAM_TSM: 128, PARAM_TSN: 64, PARAM_WPTM: 8, PARAM_WPTN: 128}
    assert run_experiments_tuned.valid_kernel6(config) is False


def test_k7_to_10_valid_config():
    config = {PARAM_WIDTH: 4, PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 8, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is True


def test_k7_to_10_tsm_not_divisible_by_wptm():
    config = {PARAM_WIDTH: 4, PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 7, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is False


def test_k7_to_10_tsm_not_divisible_by_width():
    config = {PARAM_WIDTH: 8, PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 8, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is True
    config[PARAM_TSM] = 100
    assert run_experiments_tuned.valid_kernel7_to_10(config) is False


def test_k7_to_10_tsn_not_divisible_by_width():
    config = {PARAM_WIDTH: 4, PARAM_TSM: 128, PARAM_TSN: 100, PARAM_WPTM: 8, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is False


def test_k7_to_10_width_1_is_valid():
    config = {PARAM_WIDTH: 1, PARAM_TSM: 128, PARAM_TSN: 128, PARAM_WPTM: 8, PARAM_WPTN: 8}
    assert run_experiments_tuned.valid_kernel7_to_10(config) is True
