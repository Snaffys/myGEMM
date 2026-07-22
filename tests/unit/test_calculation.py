import calculate_results_default
import calculate_results_tuned
from shared import COL_CI95, COL_MEAN, COL_STD


def test_analyze_results_tuned_mean():
    values = [100.0, 101.0, 102.0, 103.0, 104.0]
    result = calculate_results_tuned.analyze_results(values, 0)
    assert result[COL_MEAN] == 102.0


def test_analyze_results_tuned_zero_variance():
    values = [50.0, 50.0, 50.0, 50.0, 50.0]
    result = calculate_results_tuned.analyze_results(values, 0)
    assert result[COL_MEAN] == 50.0
    assert result[COL_STD] == 0.0
    assert result[COL_CI95] == 0


def test_analyze_results_default_mean():
    values = [100.0, 101.0, 102.0, 103.0, 104.0]
    result = calculate_results_default.analyze_results(values)
    assert result[COL_MEAN] == 102.0


def test_analyze_results_default_zero_variance():
    values = [50.0, 50.0, 50.0, 50.0, 50.0]
    result = calculate_results_default.analyze_results(values)
    assert result[COL_MEAN] == 50.0
    assert result[COL_STD] == 0.0
    assert result[COL_CI95] == 0


def test_process_best_configs_selects_highest_mean():
    data = {}
    config_a = (32, 8, 4, 16, 128, 128, 16, 8, 8)
    config_b = (16, 8, 4, 16, 128, 128, 16, 8, 8)
    data[(2, 8192, "myGEMM.cl", config_a)] = [750.0, 752.0]
    data[(2, 8192, "myGEMM.cl", config_b)] = [720.0, 721.0]

    best = calculate_results_tuned.process_best_configs(data)
    configs = best[(2, 8192, "myGEMM.cl")]
    assert len(configs) == 1
    assert configs[0][0] == config_a
