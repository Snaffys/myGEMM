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
    remove_outlier,
    round_mean,
    round_uncertainty,
)

INPUT_FILE = "results/raw_results_default.csv"
OUTPUT_FILE = "results/clean_results_default.csv"


def load_data(filename):
    data = {}

    with open(filename, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (int(row[COL_KERNEL]), int(row[COL_SIZE]), row[COL_IMPLEMENTATION])
            gflops = float(row[COL_GFLOPS])
            data.setdefault(key, []).append(gflops)

    return data


def analyze_results(values):
    clean_values, outliers_count = remove_outlier(values)
    print(f"Outliers: {outliers_count}/{len(values)}")

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


def write_results(data, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [COL_KERNEL, COL_SIZE, COL_IMPLEMENTATION, COL_MEAN, COL_STD, COL_CI95, COL_DAGOSTINO_P, COL_SHAPIRO_P]
        )

        for key, values in data.items():
            print(f"{key}")

            result = analyze_results(values)

            writer.writerow(
                [
                    *key,
                    result[COL_MEAN],
                    result[COL_STD],
                    result[COL_CI95],
                    round(result[COL_DAGOSTINO_P], 5),
                    round(result[COL_SHAPIRO_P], 5),
                ]
            )


def main():
    data = load_data(INPUT_FILE)
    write_results(data, OUTPUT_FILE)

    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
