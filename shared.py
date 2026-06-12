import fileinput
import math
import os
import re
import shutil
import subprocess

import numpy as np

# FILE PATHS
SETTINGS_FILE = "src/settings.h"
COMMON_FILE = "src/common.h"
SETTINGS_BACKUP = "src/settings.h.bak"
COMMON_BACKUP = "src/common.h.bak"

# EXPERIMENT SETTINGS
WARMUPS = 5
RUNS = 40
COOLDOWN_TIME_SEC = 240
SIZES = [8192, 8320]
GFLOPS_PATTERN = re.compile(r"--> *([0-9]+(?:\.[0-9]+)?) GFLOPS")
MAX_SHARED_MEM = 48 * 1024

# PARAM NAMES
PARAM_KERNEL = "KERNEL"
PARAM_TS = "TS"
PARAM_WPT = "WPT"
PARAM_WIDTH = "WIDTH"
PARAM_TSDK = "TSDK"
PARAM_TSM = "TSM"
PARAM_TSN = "TSN"
PARAM_TSK = "TSK"
PARAM_WPTM = "WPTM"
PARAM_WPTN = "WPTN"

# COLUMN NAMES
COL_KERNEL = "kernel"
COL_SIZE = "size"
COL_IMPLEMENTATION = "implementation"
COL_RUN = "run"
COL_GFLOPS = "gflops"
COL_MEAN = "mean_gflops"
COL_STD = "std_gflops"
COL_CI95 = "95ci"
COL_DAGOSTINO_P = "dagostino_p"
COL_SHAPIRO_P = "shapiro_p"


# BACKUPS
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


# MACRO HELPERS
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


# COMPILATION
def compile_project():
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make", "build", "NVFLAGS=-O3 -arch=sm_89 -Xcompiler -Wall"], check=True)


# STATISTICS
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
