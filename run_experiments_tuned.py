import csv
import fileinput
import itertools
import os
import re
import shutil
import subprocess
import time

SETTINGS_FILE = "src/settings.h"
COMMON_FILE = "src/common.h"
SETTINGS_BACKUP = "src/settings.h.bak"
COMMON_BACKUP = "src/common.h.bak"
OUTPUT_FILE = "results/raw_results_tuned.csv"

WARMUPS = 5
RUNS = 40
COOLDOWN_TIME_SEC = 240

SIZES = [8192, 8320]

GFLOPS_PATTERN = re.compile(r"--> *([0-9]+(?:\.[0-9]+)?) GFLOPS")

MAX_SHARED_MEM = 48 * 1024


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


def update_macros_in_file(filepath, macro, value):
    for line in fileinput.input(filepath, inplace=True):
        if line.startswith(f"#define {macro} "):
            print(f"#define {macro} {value}")
        else:
            print(line, end="")


def set_num_runs(num_runs):
    update_macros_in_file(COMMON_FILE, "NUM_RUNS", num_runs)


def set_size(size):
    update_macros_in_file(COMMON_FILE, "MINSIZE", f"({size})")
    update_macros_in_file(COMMON_FILE, "MAXSIZE", f"({size})")


def set_settings_file_macros(kernel, params):
    update_macros_in_file(SETTINGS_FILE, "KERNEL", kernel)
    update_macros_in_file(SETTINGS_FILE, "TS", params.get("TS", 32))
    update_macros_in_file(SETTINGS_FILE, "WPT", params.get("WPT", 8))
    update_macros_in_file(SETTINGS_FILE, "WIDTH", params.get("WIDTH", 4))
    update_macros_in_file(SETTINGS_FILE, "TSDK", params.get("TSDK", 16))
    update_macros_in_file(SETTINGS_FILE, "TSM", params.get("TSM", 128))
    update_macros_in_file(SETTINGS_FILE, "TSN", params.get("TSN", 128))
    update_macros_in_file(SETTINGS_FILE, "TSK", params.get("TSK", 16))
    update_macros_in_file(SETTINGS_FILE, "WPTM", params.get("WPTM", 8))
    update_macros_in_file(SETTINGS_FILE, "WPTN", params.get("WPTN", 8))


def shared_memory_too_large(params, kernel):
    if kernel in (1, 2, 3):
        ts = params.get("TS", 32)

        asub = ts * ts * 4
        bsub = ts * ts * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel == 4:
        ts = params.get("TS", 32)
        width = params.get("WIDTH", 1)

        asub = ts * (ts // width) * 4
        bsub = ts * (ts // width) * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel == 5:
        tsm = params["TSM"]
        tsn = params["TSN"]
        tsk = params["TSK"]

        asub = tsk * tsm * 4
        bsub = tsn * tsk * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel == 6:
        tsk = params.get("TSK", 16)
        tsm = params.get("TSM", 128)
        tsn = params.get("TSN", 128)

        asub = tsk * tsm * 4
        bsub = tsn * (tsk + 2) * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel in (7, 8):
        tsk = params.get("TSK", 16)
        tsm = params.get("TSM", 128)
        tsn = params.get("TSN", 128)

        asub = tsk * tsm * 4
        bsub = tsn * tsk * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel in (9, 10):
        tsk = params.get("TSK", 16)
        tsm = params.get("TSM", 128)
        tsn = params.get("TSN", 128)

        asub = 2 * tsk * tsm * 4
        bsub = 2 * tsn * tsk * 4
        return asub + bsub > MAX_SHARED_MEM
    else:
        return False


def valid_kernel5(params):
    ts = params["TS"]
    wpt = params["WPT"]
    tsdk = params["TSDK"]

    if ts % wpt != 0:
        return False
    if tsdk > ts:
        return False
    if ts % tsdk != 0:
        return False
    return True


def valid_kernel6(params):
    tsm = params["TSM"]
    tsn = params["TSN"]
    wptm = params["WPTM"]
    wptn = params["WPTN"]

    if tsm % wptm != 0:
        return False
    if tsn % wptn != 0:
        return False
    rtsm = tsm // wptm
    rtsn = tsn // wptn
    if rtsm <= 0 or rtsn <= 0:
        return False

    return True


def valid_kernel7_to_10(params):
    width = params["WIDTH"]
    tsm = params["TSM"]
    tsn = params["TSN"]
    wptm = params["WPTM"]
    wptn = params["WPTN"]

    if tsm % wptm != 0:
        return False
    if tsn % wptn != 0:
        return False
    if tsm % width != 0:
        return False
    if tsn % width != 0:
        return False

    return True


def kernel_param_grid(kernel):
    base = {"TS": 32, "WPT": 8, "WIDTH": 4, "TSDK": 16, "TSM": 128, "TSN": 128, "TSK": 16, "WPTM": 8, "WPTN": 8}

    grids = []

    if kernel == 1:
        if not shared_memory_too_large(dict(base), kernel):
            grids = [base]
    elif kernel == 2:
        for ts in [16, 32, 64]:
            parameters = dict(base)
            parameters["TS"] = ts
            if not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 3:
        for wpt in [2, 4, 8, 16]:
            parameters = dict(base)
            parameters["TS"] = 32
            parameters["WPT"] = wpt
            if parameters["TS"] % parameters["WPT"] == 0 and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 4:
        for width in [1, 2, 4, 8]:
            parameters = dict(base)
            parameters["WIDTH"] = width
            if not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 5:
        for ts, tsdk, wpt in itertools.product([16, 32, 64], [8, 16, 32], [2, 4, 8, 16]):
            parameters = dict(base)
            parameters.update({"TS": ts, "TSDK": tsdk, "WPT": wpt})
            if valid_kernel5(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 6:
        for tile, tsk, wpt in itertools.product([64, 128, 256], [8, 16, 32], [4, 8, 16]):
            parameters = dict(base)
            parameters.update({"TSM": tile, "TSN": tile, "TSK": tsk, "WPTM": wpt, "WPTN": wpt})
            if valid_kernel6(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 7:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [64, 128, 256], [8, 16, 32], [4, 8, 16]):
            parameters = dict(base)
            parameters.update({"WIDTH": width, "TSM": tile, "TSN": tile, "TSK": tsk, "WPTM": wpt, "WPTN": wpt})
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 8:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [64, 128, 256], [8, 16, 32], [4, 8, 16]):
            parameters = dict(base)
            parameters.update({"WIDTH": width, "TSM": tile, "TSN": tile, "TSK": tsk, "WPTM": wpt, "WPTN": wpt})
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 9:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [64, 128], [8, 16, 32], [4, 8]):
            parameters = dict(base)
            parameters.update({"WIDTH": width, "TSM": tile, "TSN": tile, "TSK": tsk, "WPTM": wpt, "WPTN": wpt})
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 10:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [128, 160], [8, 16, 32], [4, 8, 10]):
            parameters = dict(base)
            parameters.update({"WIDTH": width, "TSM": tile, "TSN": tile, "TSK": tsk, "WPTM": wpt, "WPTN": wpt})
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 11:
        grids = [base]

    return grids


def compile_project():
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make", "build", "NVFLAGS=-O3 -arch=sm_89 -Xcompiler -Wall"], check=True)


def run_program(kernel):
    result = subprocess.run(["./bin/myGEMM"], capture_output=True, text=True)
    lines = result.stdout.split("\n")

    data = {}

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
                    (implementation == "myGEMM.cu" and kernel not in [8, 9, 10])
                    or (implementation == "myGEMM.cl" and kernel == 8)
                    or (implementation in ["cuBLAS", "clBLAS"] and kernel != 11)
                ):
                    continue

                gflops_result = GFLOPS_PATTERN.search(line)
                if gflops_result:
                    data[implementation] = float(gflops_result.group(1))
                else:
                    continue

    return data


def run_config(kernel, params):
    print(f"Params: {params}")

    set_settings_file_macros(kernel, params)
    compile_project()

    print(f"Cooling down {COOLDOWN_TIME_SEC} seconds after compilation")
    time.sleep(COOLDOWN_TIME_SEC)

    print("Warmup runs")
    for _ in range(WARMUPS):
        run_program(kernel)

    rows = []

    print("Measurments")
    for run in range(1, RUNS + 1):
        print(f"Run {run}/{RUNS}")

        data = run_program(kernel)

        if not data:
            print(f"Skipping - no data for kernel {kernel}, params {params}")
            break

        for implementation, gflops in data.items():
            rows.append({"implementation": implementation, "run": run, "gflops": gflops})

    return rows


def run_experiments(result_file):
    writer = csv.writer(result_file)
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
            "run",
            "gflops",
        ]
    )

    for size in SIZES:
        print(f"Size: {size}")

        set_size(size)
        if size == 8320:
            kernels = [10, 11]
        elif size == 8192:
            kernels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for kernel in kernels:
            grid = kernel_param_grid(kernel)
            for params in grid:
                rows = run_config(kernel, params)

                for row in rows:
                    writer.writerow(
                        [
                            kernel,
                            size,
                            row["implementation"],
                            params.get("TS"),
                            params.get("WPT"),
                            params.get("WIDTH"),
                            params.get("TSDK"),
                            params.get("TSM"),
                            params.get("TSN"),
                            params.get("TSK"),
                            params.get("WPTM"),
                            params.get("WPTN"),
                            row["run"],
                            row["gflops"],
                        ]
                    )
                result_file.flush()

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
