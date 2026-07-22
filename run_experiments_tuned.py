import csv
import itertools
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
    MAX_SHARED_MEM,
    PARAM_KERNEL,
    PARAM_TS,
    PARAM_TSDK,
    PARAM_TSK,
    PARAM_TSM,
    PARAM_TSN,
    PARAM_WIDTH,
    PARAM_WPT,
    PARAM_WPTM,
    PARAM_WPTN,
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

OUTPUT_FILE = "results/raw_results_tuned.csv"


def set_settings_file_macros(kernel, params):
    update_macros_in_file(SETTINGS_FILE, PARAM_KERNEL, kernel)
    update_macros_in_file(SETTINGS_FILE, PARAM_TS, params.get(PARAM_TS, 32))
    update_macros_in_file(SETTINGS_FILE, PARAM_WPT, params.get(PARAM_WPT, 8))
    update_macros_in_file(SETTINGS_FILE, PARAM_WIDTH, params.get(PARAM_WIDTH, 4))
    update_macros_in_file(SETTINGS_FILE, PARAM_TSDK, params.get(PARAM_TSDK, 16))
    update_macros_in_file(SETTINGS_FILE, PARAM_TSM, params.get(PARAM_TSM, 128))
    update_macros_in_file(SETTINGS_FILE, PARAM_TSN, params.get(PARAM_TSN, 128))
    update_macros_in_file(SETTINGS_FILE, PARAM_TSK, params.get(PARAM_TSK, 16))
    update_macros_in_file(SETTINGS_FILE, PARAM_WPTM, params.get(PARAM_WPTM, 8))
    update_macros_in_file(SETTINGS_FILE, PARAM_WPTN, params.get(PARAM_WPTN, 8))


def shared_memory_too_large(params, kernel):
    if kernel in (1, 2, 3):
        ts = params.get(PARAM_TS, 32)

        asub = ts * ts * 4
        bsub = ts * ts * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel == 4:
        ts = params.get(PARAM_TS, 32)
        width = params.get(PARAM_WIDTH, 1)

        asub = ts * (ts // width) * 4
        bsub = ts * (ts // width) * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel == 5:
        tsm = params[PARAM_TSM]
        tsn = params[PARAM_TSN]
        tsk = params[PARAM_TSK]

        asub = tsk * tsm * 4
        bsub = tsn * tsk * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel == 6:
        tsk = params.get(PARAM_TSK, 16)
        tsm = params.get(PARAM_TSM, 128)
        tsn = params.get(PARAM_TSN, 128)

        asub = tsk * tsm * 4
        bsub = tsn * (tsk + 2) * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel in (7, 8):
        tsk = params.get(PARAM_TSK, 16)
        tsm = params.get(PARAM_TSM, 128)
        tsn = params.get(PARAM_TSN, 128)

        asub = tsk * tsm * 4
        bsub = tsn * tsk * 4
        return asub + bsub > MAX_SHARED_MEM
    elif kernel in (9, 10):
        tsk = params.get(PARAM_TSK, 16)
        tsm = params.get(PARAM_TSM, 128)
        tsn = params.get(PARAM_TSN, 128)

        asub = 2 * tsk * tsm * 4
        bsub = 2 * tsn * tsk * 4
        return asub + bsub > MAX_SHARED_MEM
    else:
        return False


def valid_kernel5(params):
    ts = params[PARAM_TS]
    wpt = params[PARAM_WPT]
    tsdk = params[PARAM_TSDK]

    if ts % wpt != 0:
        return False
    if tsdk > ts:
        return False
    if ts % tsdk != 0:
        return False
    return True


def valid_kernel6(params):
    tsm = params[PARAM_TSM]
    tsn = params[PARAM_TSN]
    wptm = params[PARAM_WPTM]
    wptn = params[PARAM_WPTN]

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
    width = params[PARAM_WIDTH]
    tsm = params[PARAM_TSM]
    tsn = params[PARAM_TSN]
    wptm = params[PARAM_WPTM]
    wptn = params[PARAM_WPTN]

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
    base = {
        PARAM_TS: 32,
        PARAM_WPT: 8,
        PARAM_WIDTH: 4,
        PARAM_TSDK: 16,
        PARAM_TSM: 128,
        PARAM_TSN: 128,
        PARAM_TSK: 16,
        PARAM_WPTM: 8,
        PARAM_WPTN: 8,
    }

    grids = []

    if kernel == 1:
        if not shared_memory_too_large(dict(base), kernel):
            grids = [base]
    elif kernel == 2:
        for ts in [16, 32, 64]:
            parameters = dict(base)
            parameters[PARAM_TS] = ts
            if not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 3:
        for wpt in [2, 4, 8, 16]:
            parameters = dict(base)
            parameters[PARAM_TS] = 32
            parameters[PARAM_WPT] = wpt
            if parameters[PARAM_TS] % parameters[PARAM_WPT] == 0 and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 4:
        for width in [1, 2, 4, 8]:
            parameters = dict(base)
            parameters[PARAM_WIDTH] = width
            if not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 5:
        for ts, tsdk, wpt in itertools.product([16, 32, 64], [8, 16, 32], [2, 4, 8, 16]):
            parameters = dict(base)
            parameters.update({PARAM_TS: ts, PARAM_TSDK: tsdk, PARAM_WPT: wpt})
            if valid_kernel5(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 6:
        for tile, tsk, wpt in itertools.product([64, 128, 256], [8, 16, 32], [4, 8, 16]):
            parameters = dict(base)
            parameters.update({PARAM_TSM: tile, PARAM_TSN: tile, PARAM_TSK: tsk, PARAM_WPTM: wpt, PARAM_WPTN: wpt})
            if valid_kernel6(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 7:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [64, 128, 256], [8, 16, 32], [4, 8, 16]):
            parameters = dict(base)
            parameters.update(
                {PARAM_WIDTH: width, PARAM_TSM: tile, PARAM_TSN: tile, PARAM_TSK: tsk, PARAM_WPTM: wpt, PARAM_WPTN: wpt}
            )
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 8:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [64, 128, 256], [8, 16, 32], [4, 8, 16]):
            parameters = dict(base)
            parameters.update(
                {PARAM_WIDTH: width, PARAM_TSM: tile, PARAM_TSN: tile, PARAM_TSK: tsk, PARAM_WPTM: wpt, PARAM_WPTN: wpt}
            )
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 9:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [64, 128], [8, 16, 32], [4, 8]):
            parameters = dict(base)
            parameters.update(
                {PARAM_WIDTH: width, PARAM_TSM: tile, PARAM_TSN: tile, PARAM_TSK: tsk, PARAM_WPTM: wpt, PARAM_WPTN: wpt}
            )
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 10:
        for width, tile, tsk, wpt in itertools.product([2, 4, 8], [128, 160], [8, 16, 32], [4, 8, 10]):
            parameters = dict(base)
            parameters.update(
                {PARAM_WIDTH: width, PARAM_TSM: tile, PARAM_TSN: tile, PARAM_TSK: tsk, PARAM_WPTM: wpt, PARAM_WPTN: wpt}
            )
            if valid_kernel7_to_10(parameters) and not shared_memory_too_large(parameters, kernel):
                grids.append(parameters)
    elif kernel == 11:
        grids = [base]

    return grids


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
            rows.append({COL_IMPLEMENTATION: implementation, COL_RUN: run, COL_GFLOPS: gflops})

    return rows


def run_experiments(result_file):
    writer = csv.writer(result_file)
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
            COL_RUN,
            COL_GFLOPS,
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
                            row[COL_IMPLEMENTATION],
                            params.get(PARAM_TS),
                            params.get(PARAM_WPT),
                            params.get(PARAM_WIDTH),
                            params.get(PARAM_TSDK),
                            params.get(PARAM_TSM),
                            params.get(PARAM_TSN),
                            params.get(PARAM_TSK),
                            params.get(PARAM_WPTM),
                            params.get(PARAM_WPTN),
                            row[COL_RUN],
                            row[COL_GFLOPS],
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
