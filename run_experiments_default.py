import csv
import fileinput
import os
import re
import shutil
import subprocess
import time

SETTINGS_FILE = "src/settings.h"
COMMON_FILE = "src/common.h"
SETTINGS_BACKUP = "src/settings.h.bak"
COMMON_BACKUP = "src/common.h.bak"
OUTPUT_FILE = "results/raw_results_default.csv"

WARMUPS = 5
RUNS = 40
COOLDOWN_TIME_SEC = 240

SIZES = [8192, 8320]

GFLOPS_PATTERN = re.compile(r"--> *([0-9]+(?:\.[0-9]+)?) GFLOPS")


def create_backup():
    shutil.copy(COMMON_FILE, COMMON_BACKUP)
    shutil.copy(SETTINGS_FILE, SETTINGS_BACKUP)


def restore_files():
    if os.path.exists(COMMON_BACKUP):
        shutil.copy(COMMON_BACKUP, COMMON_FILE)
        os.remove(COMMON_BACKUP)
    if os.path.exists(SETTINGS_BACKUP):
        shutil.copy(SETTINGS_BACKUP, SETTINGS_FILE)
        os.remove(SETTINGS_BACKUP)


def set_num_runs(num_runs):
    for line in fileinput.input(COMMON_FILE, inplace=True):
        if line.startswith("#define NUM_RUNS"):
            print(f"#define NUM_RUNS {num_runs}")
        else:
            print(line, end="")


def set_size(size):
    for line in fileinput.input(COMMON_FILE, inplace=True):
        if line.startswith("#define MINSIZE"):
            print(f"#define MINSIZE ({size})")
        elif line.startswith("#define MAXSIZE"):
            print(f"#define MAXSIZE ({size})")
        else:
            print(line, end="")


def set_kernel(kernel):
    for line in fileinput.input(SETTINGS_FILE, inplace=True):
        if line.startswith("#define KERNEL"):
            print(f"#define KERNEL {kernel}")
        else:
            print(line, end="")


def compile_project():
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make", "build", "NVFLAGS=-O3 -arch=sm_89 -Xcompiler -Wall"], check=True)


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

                data[current_size][implementation] = {"gflops": gflops}

    return data


def run_kernel(writer, result_file, size, kernel):
    print(f"Kernel: {kernel}")

    set_kernel(kernel)
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
            gflops = data[size][implementation]["gflops"]
            writer.writerow([kernel, size, implementation, run, gflops])

        result_file.flush()


def run_experiments(result_file):
    writer = csv.writer(result_file)
    writer.writerow(["kernel", "size", "implementation", "run", "gflops"])

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
