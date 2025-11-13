#!/usr/bin/env python3
import os
import subprocess
import sys

CXX_COMPILER = "g++"
CXX_FLAGS = ["-std=c++20", "-O2", "-Wall", "-Wextra"]

EIGEN3_INCLUDE_DIR = os.path.join("/opt/homebrew/Cellar/eigen/3.4.0_1", "include")
INCLUDES = [EIGEN3_INCLUDE_DIR]

EXAMPLES_SRCS = [
    "src/examples/ising_model.h",
    "src/examples/measure.h",
    "src/examples/utils.h",
    "src/examples/example_measure.cpp",
    "src/examples/hard_sphere.h",
    "src/wang_landau.h",
]

SRCS = [*EXAMPLES_SRCS]

CXX_FILE = "src/main.cpp"
OUTPUT_FILE = "wang_landau"

def cc():
    if not should_rebuild():
        print(f"{OUTPUT_FILE} is up to date.")
    else:
        cmd = [CXX_COMPILER, *CXX_FLAGS, "-I", *INCLUDES, "-o", OUTPUT_FILE, CXX_FILE]
        subprocess.run([CXX_COMPILER, *CXX_FLAGS, "-I", *INCLUDES, "-o", OUTPUT_FILE, CXX_FILE], check=True)

def should_rebuild():
    if not os.path.exists(OUTPUT_FILE):
        return True

    output_mtime = os.path.getmtime(OUTPUT_FILE)
    source_files = [CXX_FILE] + SRCS
    for src in source_files:
        if os.path.getmtime(src) > output_mtime:
            return True
    return False

if __name__ == "__main__":
    try:
        cc()
    except subprocess.CalledProcessError as e:
        sys.exit(1)