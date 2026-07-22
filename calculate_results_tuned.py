import csv

import numpy as np
from scipy import stats

from shared import (
    COL_CI95,
    COL_DAGOSTINO_P,
    COL_GFLOPS,
    COL_IMPLEMENTATION,
    COL_KERNEL,
    COL_MEAN,
    COL_SHAPIRO_P,
    COL_SIZE,
    COL_STD,
    PARAM_TS,
    PARAM_TSDK,
    PARAM_TSK,
    PARAM_TSM,
    PARAM_TSN,
    PARAM_WIDTH,
    PARAM_WPT,
    PARAM_WPTM,
    PARAM_WPTN,
    remove_outlier,
    round_mean,
    round_uncertainty,
)

INPUT_FILE = "results/raw_results_tuned.csv"
OUTPUT_FILE = "results/clean_results_tuned.csv"


def load_data(filename):
    data = {}

    with open(filename, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            config = (
                int(row[PARAM_TS]),
                int(row[PARAM_WPT]),
                int(row[PARAM_WIDTH]),
                int(row[PARAM_TSDK]),
                int(row[PARAM_TSM]),
                int(row[PARAM_TSN]),
                int(row[PARAM_TSK]),
                int(row[PARAM_WPTM]),
                int(row[PARAM_WPTN]),
            )
            key = (int(row[COL_KERNEL]), int(row[COL_SIZE]), row[COL_IMPLEMENTATION], config)
            gflops = float(row[COL_GFLOPS])
            data.setdefault(key, []).append(gflops)

    return data


def analyze_results(clean_values, outliers_count):
    print(f"Outliers: {outliers_count}")

    _, dagostino_p = stats.normaltest(clean_values)
    _, shapiro_p = stats.shapiro(clean_values)
    print(f"D'Agostino p: {dagostino_p}")
    print(f"Shapiro p: {shapiro_p}")
    if dagostino_p <= 0.05 and shapiro_p <= 0.05:
        print("Distribution is not normal!")

    mean = np.mean(clean_values)
    std = np.std(clean_values, ddof=1)
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    ci95 = stats.t.ppf(0.975, df=len(clean_values) - 1) * stats.sem(clean_values)

    ci95_rounded = round_uncertainty(ci95)
    mean_rounded = round_mean(mean, ci95_rounded)
    std_rounded = round(std, 2)

    return {
        COL_MEAN: mean_rounded,
        COL_STD: std_rounded,
        COL_CI95: ci95_rounded,
        COL_DAGOSTINO_P: dagostino_p,
        COL_SHAPIRO_P: shapiro_p,
    }


def process_best_configs(data):
    configs_group = {}

    for (kernel, size, implementation, config), values in data.items():
        key = (kernel, size, implementation)
        configs_group.setdefault(key, []).append((config, values))

    best = {}

    for group_key, configs in configs_group.items():
        best_mean = -1
        best_std = None
        best_configs = []

        for config, values in configs:
            clean_values, outliers_count = remove_outlier(values)
            mean = np.mean(clean_values)
            std = np.std(clean_values, ddof=1)

            if mean > best_mean:
                best_mean = mean
                best_std = std
                best_configs = [(config, clean_values, outliers_count)]
            elif mean == best_mean:
                if std < best_std:
                    best_std = std
                    best_configs = [(config, clean_values, outliers_count)]
                elif std == best_std:
                    best_configs.append((config, clean_values, outliers_count))

        best[group_key] = best_configs

    return best


def write_results(best_data, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                COL_KERNEL,
                COL_SIZE,
                COL_IMPLEMENTATION,
                PARAM_TS,
                PARAM_WPT,
                PARAM_WIDTH,
                PARAM_TSDK,
                PARAM_TSM,
                PARAM_TSN,
                PARAM_TSK,
                PARAM_WPTM,
                PARAM_WPTN,
                COL_MEAN,
                COL_STD,
                COL_CI95,
                COL_DAGOSTINO_P,
                COL_SHAPIRO_P,
            ]
        )

        for key, configs in best_data.items():
            print(f"{key}")
            for config, clean_values, outliers_count in configs:
                print(*config)

                result = analyze_results(clean_values, outliers_count)

                writer.writerow(
                    [
                        *key,
                        *config,
                        result[COL_MEAN],
                        result[COL_STD],
                        result[COL_CI95],
                        round(result[COL_DAGOSTINO_P], 5),
                        round(result[COL_SHAPIRO_P], 5),
                    ]
                )


def main():
    data = load_data(INPUT_FILE)
    best = process_best_configs(data)
    write_results(best, OUTPUT_FILE)

    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
