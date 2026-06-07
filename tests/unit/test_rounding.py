def test_round_uncertainty():
    assert calculate_results_original.round_uncertainty(0.0456) == 0.05


def test_round_uncertainty_starting_with_1():
    assert calculate_results_original.round_uncertainty(0.00123) == 0.0012


def test_round_mean_small_uncertainty():
    assert calculate_results_original.round_mean(0.1234, 0.005) == 0.123


def test_round_mean_large_uncertainty():
    assert calculate_results_original.round_mean(1234.56, 700) == 1200
