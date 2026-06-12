import csv
import subprocess
import time

from shared import (
    COL_GFLOPS,
    COL_IMPLEMENTATION,
    COL_KERNEL,
    COL_RUN,
    COL_SIZE,
    COOLDOWN_TIME_SEC,
    GFLOPS_PATTERN,
    PARAM_KERNEL,
    RUNS,
    SETTINGS_FILE,
    SIZES,
    WARMUPS,
    compile_project,
    create_backup,
    restore_files,
    set_num_runs,
    set_size,
    update_macros_in_file,
)

OUTPUT_FILE = "results/raw_results_default.csv"


def run_program(current_kernel, current_size):
    result = subprocess.run(["./bin/myGEMM"], capture_output=True, text=True)
    lines = result.stdout.split("\n")

    data = {current_size: {}}

    for line in lines:
        if "GFLOPS" in line:
            implementation = None
            if "cuBLAS" in line:
                implementation = "cuBLAS"
            elif "clBLAS" in line:
                implementation = "clBLAS"
            elif "myGEMM.cu" in line:
                implementation = "myGEMM.cu"
            elif "myGEMM.cl" in line:
                implementation = "myGEMM.cl"

            if implementation:
                if (
                    (implementation == "myGEMM.cu" and current_kernel not in [8, 9, 10])
                    or (implementation == "myGEMM.cl" and current_kernel == 8)
                    or (implementation in ["cuBLAS", "clBLAS"] and current_kernel != 11)
                ):
                    continue

                gflops_result = GFLOPS_PATTERN.search(line)
                if gflops_result:
                    gflops = float(gflops_result.group(1))
                else:
                    continue

                data[current_size][implementation] = {COL_GFLOPS: gflops}

    return data


def run_kernel(writer, result_file, size, kernel):
    print(f"Kernel: {kernel}")

    update_macros_in_file(SETTINGS_FILE, PARAM_KERNEL, kernel)
    compile_project()

    print(f"Cooling down {COOLDOWN_TIME_SEC} seconds after compilation")
    time.sleep(COOLDOWN_TIME_SEC)

    print("Warmup runs")
    for _ in range(WARMUPS):
        run_program(kernel, size)

    print("Measurements")
    for run in range(1, RUNS + 1):
        print(f"Run {run}/{RUNS}")

        data = run_program(kernel, size)

        if not data[size]:
            print(f"Skipping - no data for kernel {kernel}, size {size}")
            break

        for implementation in data[size]:
            gflops = data[size][implementation][COL_GFLOPS]
            writer.writerow([kernel, size, implementation, run, gflops])

        result_file.flush()


def run_experiments(result_file):
    writer = csv.writer(result_file)
    writer.writerow([COL_KERNEL, COL_SIZE, COL_IMPLEMENTATION, COL_RUN, COL_GFLOPS])

    for size in SIZES:
        print(f"Size: {size}")

        set_size(size)
        if size == 8320:
            kernels = [10, 11]
        elif size == 8192:
            kernels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for kernel in kernels:
            run_kernel(writer, result_file, size, kernel)

        print(f"Cooling down {COOLDOWN_TIME_SEC} seconds between sizes")
        time.sleep(COOLDOWN_TIME_SEC)


def main():
    create_backup()

    try:
        set_num_runs(1)

        with open(OUTPUT_FILE, "w", newline="") as result_file:
            run_experiments(result_file)

        print(f"Results saved to {OUTPUT_FILE}")
    finally:
        restore_files()


if __name__ == "__main__":
    main()
