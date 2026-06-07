def test_keep_normal_data():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    cleaned, outliers = calculate_results_original.remove_outlier(values)
    assert outliers == 0
    assert list(cleaned) == values


def test_remove_single_outlier():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    cleaned, outliers = calculate_results_original.remove_outlier(values)
    assert outliers == 1
    assert 100 not in cleaned


def test_keep_all_when_many_outliers():
    values = [1, 100, 101, 102, 103, 104, 200]
    cleaned, outliers = calculate_results_original.remove_outlier(values)
    assert outliers > 1
    assert list(cleaned) == values
