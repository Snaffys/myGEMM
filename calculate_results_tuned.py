import csv
import math

import numpy as np
from scipy import stats

INPUT_FILE = "results/raw_results_tuned.csv"
OUTPUT_FILE = "results/clean_results_tuned.csv"


def load_data(filename):
    data = {}

    with open(filename, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            config = (
                int(row["TS"]),
                int(row["WPT"]),
                int(row["WIDTH"]),
                int(row["TSDK"]),
                int(row["TSM"]),
                int(row["TSN"]),
                int(row["TSK"]),
                int(row["WPTM"]),
                int(row["WPTN"]),
            )
            key = (int(row["kernel"]), int(row["size"]), row["implementation"], config)
            gflops = float(row["gflops"])
            data.setdefault(key, []).append(gflops)

    return data


def remove_outlier(values):
    values = np.array(values)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (values >= lower) & (values <= upper)
    cleaned = values[mask]
    outliers_count = len(values) - len(cleaned)

    if outliers_count <= 1:
        return cleaned, outliers_count
    else:
        return values, outliers_count


def round_uncertainty(value):
    if value == 0:
        return 0

    order = int(math.floor(math.log10(abs(value))))
    first_digit = int(value / 10**order)

    if first_digit == 1:
        round_digits = order - 1
    else:
        round_digits = order

    return round(value, -round_digits)


def round_mean(value, uncertainty):
    if uncertainty == 0:
        return round(value)

    digits = int(math.floor(math.log10(abs(uncertainty))))
    return round(value, -digits)


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
        "mean_rounded": mean_rounded,
        "std_rounded": std_rounded,
        "ci95_rounded": ci95_rounded,
        "dagostino_p": dagostino_p,
        "shapiro_p": shapiro_p,
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
                "kernel",
                "size",
                "implementation",
                "TS",
                "WPT",
                "WIDTH",
                "TSDK",
                "TSM",
                "TSN",
                "TSK",
                "WPTM",
                "WPTN",
                "mean_gflops",
                "std_gflops",
                "95ci",
                "dagostino_p",
                "shapiro_p",
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
                        result["mean_rounded"],
                        result["std_rounded"],
                        result["ci95_rounded"],
                        round(result["dagostino_p"], 5),
                        round(result["shapiro_p"], 5),
                    ]
                )


def main():
    data = load_data(INPUT_FILE)
    best = process_best_configs(data)
    write_results(best, OUTPUT_FILE)

    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
