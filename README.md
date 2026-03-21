
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

- **Matrix sizes tested:** `8192x8192x8192`, `8320x8320x8320`
- **Kernel executions:** Each kernel was executed **40 times per matrix size** to ensure stable statistics
- **Warm-up runs:** A few preliminary runs were performed before measurements to stabilize GPU performance
- **Pauses between runs:** Introduced to reduce thermal throttling and other system effects
- **Performance metric:** GFLOPS (Giga Floating Point Operations per Second)
- **Measured statistics:**
  - Mean GFLOPS
  - Standard deviation (std)
  - 95% confidence interval (CI95)
  - p-values from D'Agostino and Shapiro tests for normality

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

### Matrix Size: 8192x8192x8192

| Implementation | Mean GFLOPS | Std GFLOPS | 95% CI | Dagostino p | Shapiro p |
|----------------|-------------|------------|--------|-------------|-----------|
| cuBLAS         | 6480        | 173        | 60     | 0.42        | 0.48      |
| clBLAS         | 3460        | 77         | 20     | 0.003       | 0.00001   |
| myGEMM1 (cl)   | 714         | 2          | 1      | 0.71        | 0.67      |
| myGEMM2 (cl)   | 1022        | 2          | 1      | 0.81        | 0.89      |
| myGEMM3 (cl)   | 2428        | 7          | 2      | 0.24        | 0.073     |
| myGEMM4 (cl)   | 2065        | 16         | 6      | 0.10        | 0.0009    |
| myGEMM5 (cl)   | 1324        | 6          | 2      | 0.17        | 0.035     |
| myGEMM6 (cl)   | 4910        | 135        | 40     | 0.0         | 0.00009   |
| myGEMM7 (cl)   | 5080        | 42         | 14     | 0.38        | 0.15      |
| myGEMM8 (cu)   | 5400        | 99         | 30     | 0.35        | 0.073     |
| myGEMM9 (cu)   | 5310        | 112        | 40     | 0.14        | 0.0053    |
| myGEMM9 (cl)   | 5130        | 39         | 13     | 0.17        | 0.25      |
| myGEMM10 (cu)  | 5230        | 29         | 11     | 0.11        | 0.0016    |
| myGEMM10 (cl)  | 4830        | 32         | 11     | 0.053       | 0.0011    |
| myGEMM11 (cl)  | 4090        | 75         | 20     | 0.0         | 0.0       |

---

### Matrix Size: 8320x8320x8320

| Implementation | Mean GFLOPS | Std GFLOPS | 95% CI | Dagostino p | Shapiro p |
|----------------|------------ |------------|--------|-------------|-----------|
| cuBLAS         | 6510        | 138        | 40     | 0.86        | 0.024     |
| clBLAS         | 3520        | 74         | 20     | 0.0006      | 0.00003   |
| myGEMM10 (cu)  | 5180        | 106        | 30     | 0.065       | 0.010     |
| myGEMM10 (cl)  | 4880        | 41         | 13     | 0.16        | 0.094     |
| myGEMM11 (cl)  | 4020        | 66         | 20     | 0.00001     | 0.00001   |

---

## Charts

Charts of GFLOPS performance per kernel and matrix size are available in the `charts/` folder:

![8192](charts/performance_8192.png) 
![8320](charts/performance_8320.png)

---

## Notes

- Each kernel was executed with all other applications closed, laptop in **performance mode** and continuously **plugged in**, to reduce interference
- Multiple runs ensure statistical stability and allow computation of meaningful standard deviations and confidence intervals
- Some implementations (clBLAS, myGEMM6, myGEMM11) exhibit non-normal performance distributions, likely due to hardware-specific optimizations, register pressure and compiler optimizations
- All experiments were performed using default kernel parameters, without manual tuning for this GPU
