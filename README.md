
Exploring the performance of SGEMM in OpenCL on NVIDIA GPUs
=============

Date: 31-Oct-2014 - 07-Nov-2014

Author: Cedric Nugteren, SURFsara (http://www.surfsara.nl)

This repository contains multiple OpenCL implementations of single-precision generalised matrix-multiplication (SGEMM) tuned for an NVIDIA Tesla K40m GPU. The different versions (named myGEMM) are part of a step-by-step tutorial, in which each step adds a new optimisation. The different steps and the details of the OpenCL kernel codes are all explained in depth at http://www.cedricnugteren.nl/tutorial.php.

The OpenCL kernels can be used natively using the OpenCL framework. However, there is also a header-file included which converts the OpenCL kernels into CUDA syntax. This allows the same code to be tested through the CUDA-toolchain.

Apart from the OpenCL kernel codes, this repository contains fully working host code, including a loop over different matrix sizes and different BLAS libraries. It contains code to run NVIDIA's cuBLAS as a reference and the open-source clBlas library.

Pre-requisites:
* A C++ compiler (tested with GCC and ICC)
* The CUDA toolkit and NVCC compiler (tested with version 6.5)
* OpenCL headers and libraries (part of the CUDA toolkit)

Requirements to run the performance and correctness comparisons:
* The cuBLAS library (part of the CUDA toolkit, tested version 6.5)
* The open-source clBlas library (tested 2.2.0)

Usage
=============

*	Compile the code:

		make build

	Compiles the benchmarking infrastructure and the myGEMM kernels. Make sure there is a "bin" and "obj" directory available. Note that you might have to edit the Makefile to set the proper locations of the CUDA and OpenCL installations on your system.

*	Run the code:

		make run

	This runs the code for matrices ranging from MINSIZE to MAXSIZE (defined in src/common.h). It will run cuBLAS, clBlas, and the CUDA and OpenCL versions of the myGEMM kernels. The particular kernel to be executed is defined using the KERNEL keyword in src/settings.h. This file also contains other settings you might want to modify for your particular GPU.

*	Inspect the code:

		make inspect

	This generates all kinds of assembly-like versions of the CUDA kernels in the "bin" subdirectory. It also prints out statistics of the kernels such as the register usage.

Minimal working example
=============

Additionally, we supply the minimal.cpp file in the 'extra' directory. This file is a self-contained minimal working example (MWE) of the most basic SGEMM kernel (myGEMM1). This can be useful if you don't want to deal with Makefiles or don't have the CUDA, cuBLAS, or clBlas installed. Note that minimal.cpp misses some features compared to the main code, but we believe that it can nevertheless be a good starting point if you want to integrate myGEMM into your own code.

The code can be compiled using a regular C++ compiler and only requires OpenCL installed. Example compilation from the root folder:

	g++ -O3 -Wall -I/path/to/opencl/include extra/minimal.cpp -o bin/minimal -lOpenCL

Be aware that the minimal working example does not:
*	Iterate over multiple matrix sizes
*	Compare performance with cuBLAS or clBlas
*	Check for correctness of the results
*	Check for OpenCL errors
*	Load a kernel-file from disk, instead it is embedded as a string

###################################################

# GEMM Optimization Experiments

This repository is based on the OpenCL SGEMM tuning tutorial by Cedric Nugteren:
https://cnugteren.github.io/tutorial/pages/page1.html

The goal of this work is to **study the step-by-step optimization of single-precision matrix multiplication (SGEMM) on a GPU**, and to conduct experiments evaluating how different optimization techniques affect performance.

---

## Experimental Setup

### Hardware

- **CPU:** AMD Ryzen 7 7435HS (8 cores / 16 threads)
- **RAM:** 16 GB
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
  - GPU Memory: 8 GB
  - Compute Capability: 8.9

### Software

- **OS:** Manjaro Linux
- **GPU Driver:** 590.48.01
- **CUDA Toolkit:** 13.1
- **NVCC Compiler:** 13.1.115
- **OpenCL Platform:** OpenCL 3.0
- **Compiler:** GCC 15.2.1

### Libraries

- CUDA
- OpenCL 
- cuBLAS
- clBLAS

---

## Benchmark Configuration and Measurement Methodology

To obtain statistically reliable measurements of SGEMM performance, the following procedure was used:

- **Matrix sizes tested:** `1024x1024`, `2048x2048`, `4096x4096`
- **Kernel executions:** Each kernel was executed **40 times per matrix size** to ensure stable statistics
- **Warm-up runs:** A few preliminary runs were performed before measurements to stabilize GPU performance
- **Pauses between runs:** Introduced to reduce thermal throttling and other system effects
- **Performance metric:** GFLOPS (Giga Floating Point Operations per Second)
- **Measured statistics:**
  - Mean GFLOPS
  - Standard deviation (std)
  - 95% confidence interval (CI95)

---

## SGEMM Optimization Steps

The tutorial progressively optimizes the SGEMM kernel. Each kernel corresponds to a specific optimization step:

| Kernel | Optimization                                  |
|--------|-----------------------------------------------|
| 1      | Naive implementation                          |
| 2      | Tiling in the local memory                    |
| 3      | More work per thread                          |
| 4      | Wider data-types                              |
| 5      | Transposed input matrix and rectangular tiles |
| 6      | 2D register blocking                          |
| 7      | Wider loads with register blocking            |
| 8      | CUDA and Kepler-specific optimizations        |
| 9      | Pre-fetching                                  |
| 10     | Incomplete tiles and arbitrary matrix sizes   |
| 11     | Evaluation of the clBLAS SGEMM kernel         |

---

## Performance Tables

### Matrix Size: 1024x1024x1024

| Implementation | Mean GFLOPS | Std GFLOPS | 95% CI |
|----------------|-------------|------------|--------|
| cuBLAS         | 5853.07     | 54.32      | 17.37  |
| clBLAS         | 1695.06     | 22.77      | 7.28   |
| myGEMM1 (cl)   | 706.01      | 7.50       | 2.40   |
| myGEMM2 (cl)   | 903.51      | 11.17      | 3.57   |
| myGEMM3 (cl)   | 2606.92     | 12.83      | 4.16   |
| myGEMM4 (cl)   | 2019.59     | 25.43      | 8.13   |
| myGEMM5 (cl)   | 1105.68     | 7.87       | 2.52   |
| myGEMM6 (cl)   | 3780.80     | 37.02      | 11.84  |
| myGEMM7 (cl)   | 3873.90     | 12.28      | 3.98   |
| myGEMM8 (cu)   | 4100.88     | 10.43      | 3.38   |
| myGEMM9 (cu)   | 4043.26     | 9.49       | 3.03   |
| myGEMM9 (cl)   | 3858.99     | 19.21      | 6.14   |
| myGEMM10 (cu)  | 3749.39     | 12.61      | 4.03   |
| myGEMM10 (cl)  | 3377.90     | 38.07      | 12.17  |
| myGEMM11 (cl)  | 3871.36     | 103.04     | 32.95  |

---

### Matrix Size: 2048x2048x2048

| Implementation | Mean GFLOPS | Std GFLOPS | 95% CI |
|----------------|-------------|------------|--------|
| cuBLAS         | 6833.13     | 66.75      | 21.35  |
| clBLAS         | 2158.19     | 10.32      | 3.30   |
| myGEMM1 (cl)   | 904.22      | 51.90      | 16.60  |
| myGEMM2 (cl)   | 1182.25     | 96.70      | 30.93  |
| myGEMM3 (cl)   | 2746.95     | 147.36     | 47.13  |
| myGEMM4 (cl)   | 2226.46     | 169.25     | 54.13  |
| myGEMM5 (cl)   | 1262.94     | 79.46      | 25.41  |
| myGEMM6 (cl)   | 4573.56     | 149.99     | 47.97  |
| myGEMM7 (cl)   | 4671.09     | 215.89     | 69.05  |
| myGEMM8 (cu)   | 4898.26     | 24.11      | 7.71   |
| myGEMM9 (cu)   | 4799.70     | 31.34      | 10.02  |
| myGEMM9 (cl)   | 4636.98     | 193.65     | 61.93  |
| myGEMM10 (cu)  | 4280.77     | 18.49      | 5.91   |
| myGEMM10 (cl)  | 4183.27     | 151.41     | 48.42  |
| myGEMM11 (cl)  | 4405.14     | 131.73     | 42.13  |

---

### Matrix Size: 4096x4096x4096

| Implementation | Mean GFLOPS | Std GFLOPS | 95% CI |
|----------------|-------------|------------|--------|
| cuBLAS         | 7767.26     | 714.88     | 228.63 |
| clBLAS         | 3344.76     | 198.41     | 63.45  |
| myGEMM1 (cl)   | 718.28      | 7.76       | 2.48   |
| myGEMM2 (cl)   | 1052.02     | 4.81       | 1.54   |
| myGEMM3 (cl)   | 2481.00     | 109.65     | 35.07  |
| myGEMM4 (cl)   | 2094.12     | 77.30      | 24.72  |
| myGEMM5 (cl)   | 1325.65     | 17.53      | 5.61   |
| myGEMM6 (cl)   | 5715.48     | 553.62     | 177.06 |
| myGEMM7 (cl)   | 5772.22     | 472.96     | 151.26 |
| myGEMM8 (cu)   | 5584.59     | 308.41     | 98.64  |
| myGEMM9 (cu)   | 5568.71     | 344.22     | 111.58 |
| myGEMM9 (cl)   | 5825.66     | 505.28     | 161.60 |
| myGEMM10 (cu)  | 5277.10     | 266.95     | 85.37  |
| myGEMM10 (cl)  | 5303.34     | 359.76     | 115.06 |
| myGEMM11 (cl)  | 4702.04     | 278.53     | 89.08  |

---

## Charts

Charts of GFLOPS performance per kernel and matrix size are available in the `charts/` folder:

![1024](charts/performance_1024.png) 
![2048](charts/performance_2048.png)
![4096](charts/performance_4096.png)

---

## Notes

- Each kernel was executed with all other applications closed, GPU in **performance mode**, and laptop continuously **plugged in**, to reduce interference
- Multiple runs ensure statistical stability and allow computation of meaningful standard deviations and confidence intervals
