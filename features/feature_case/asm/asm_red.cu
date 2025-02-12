// ====------ asm_red.cu ---------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include "cuda.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

__global__ void relaxed_add_kernel(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float value = data[idx];

    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;"
                 :
                 : "l"(data), "f"(value));
  }
}

__global__ void relaxed_or_kernel(int *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int value = data[idx];

    asm volatile("red.relaxed.gpu.global.or.b32 [%0], %1;"
                 :
                 : "l"(data), "r"(value));
  }
}

__global__ void relaxed_xor_kernel(int *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int value = data[idx];

    asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;"
                 :
                 : "l"(data), "r"(value));
  }
}

__global__ void relaxed_and_kernel(int *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int value = data[idx];

    asm volatile("red.relaxed.gpu.global.and.b32 [%0], %1;"
                 :
                 : "l"(data), "r"(value));
  }
}

__global__ void relaxed_max_kernel(int *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int value = data[idx];

    asm volatile("red.relaxed.gpu.global.max.s32 [%0], %1;"
                 :
                 : "l"(data), "r"(value));
  }
}

__global__ void relaxed_min_kernel(int *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int value = data[idx];

    asm volatile("red.relaxed.gpu.global.min.s32 [%0], %1;"
                 :
                 : "l"(data), "r"(value));
  }
}

void relaxed_add_kernel_test(void) {
  const int size = 100;
  float *d_data, h_data[size];

  // Initialize host data
  for (int i = 0; i < size; i++) {
    h_data[i] = static_cast<float>(i);
  }

  // Allocate device memory
  cudaMalloc(&d_data, size * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

  relaxed_add_kernel<<<1, size>>>(d_data, size);
  cudaDeviceSynchronize();
  // Copy results back to host
  cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);

  if (h_data[0] != 4950) {
    std::cout << "add value: " << h_data[0] << std::endl;
    std::cout << "relaxed_add_kernel_test run failed!\n";
    exit(-1);
  }
  std::cout << "relaxed_add_kernel_test run passed!\n";
}

void relaxed_or_kernel_test(void) {
  const int size = 50;
  int *d_data, h_data[size];

  // Initialize host data
  for (int i = 0; i < size; i++) {
    h_data[i] = 0xF;
  }

  // Allocate device memory
  cudaMalloc(&d_data, size * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

  relaxed_or_kernel<<<1, size>>>(d_data, size);
  cudaDeviceSynchronize();
  // Copy results back to host
  cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);

  if (h_data[0] != 0xF) {
    std::cout << "or value: " << h_data[0] << std::endl;
    std::cout << "relaxed_or_kernel_test run failed!\n";
    exit(-1);
  }
  std::cout << "relaxed_or_kernel_test run passed!\n";
}

void relaxed_xor_kernel_test(void) {
  const int size = 2;
  int *d_data, h_data[size];

  // Initialize host data
  for (int i = 0; i < size; i++) {
    h_data[i] = 0xFFFFFFFF;
  }

  // Allocate device memory
  cudaMalloc(&d_data, size * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

  relaxed_xor_kernel<<<1, size>>>(d_data, size);
  cudaDeviceSynchronize();
  // Copy results back to host
  cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);

  if (h_data[0] != 0x0) {
    std::cout << "xor value: " << h_data[0] << std::endl;
    std::cout << "relaxed_xor_kernel_test run failed!\n";
    exit(-1);
  }
  std::cout << "relaxed_xor_kernel_test run passed!\n";
}

void relaxed_and_kernel_test(void) {
  const int size = 32;
  int *d_data, h_data[size];

  // Initialize host data
  for (int i = 0; i < size; i++) {
    h_data[i] = 0xF;
  }

  // Allocate device memory
  cudaMalloc(&d_data, size * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

  relaxed_and_kernel<<<1, size>>>(d_data, size);
  cudaDeviceSynchronize();
  // Copy results back to host
  cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);

  if (h_data[0] != 0xF) {
    std::cout << "and value: " << h_data[0] << std::endl;
    std::cout << "relaxed_and_kernel_test run failed!\n";
    exit(-1);
  }
  std::cout << "relaxed_and_kernel_test run passed!\n";
}

void relaxed_max_kernel_test(void) {
  const int size = 100;
  int *d_data, h_data[size];

  // Initialize host data
  for (int i = 0; i < size; i++) {
    h_data[i] = i;
  }

  // Allocate device memory
  cudaMalloc(&d_data, size * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

  relaxed_max_kernel<<<1, size>>>(d_data, size);
  cudaDeviceSynchronize();
  // Copy results back to host
  cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);

  if (h_data[0] != 99) {
    std::cout << "max value: " << h_data[0] << std::endl;
    std::cout << "relaxed_max_kernel_test run failed!\n";
    exit(-1);
  }
  std::cout << "relaxed_max_kernel_test run passed!\n";
}

void relaxed_min_kernel_test(void) {
  const int size = 100;
  int *d_data, h_data[size];

  // Initialize host data
  for (int i = 0; i < size; i++) {
    h_data[i] = i;
  }

  // Allocate device memory
  cudaMalloc(&d_data, size * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

  relaxed_min_kernel<<<1, size>>>(d_data, size);
  cudaDeviceSynchronize();
  // Copy results back to host
  cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_data);

  if (h_data[0] != 0) {
    std::cout << "min value: " << h_data[0] << std::endl;
    std::cout << "relaxed_min_kernel_test run failed!\n";
    exit(-1);
  }
  std::cout << "relaxed_min_kernel_test run passed!\n";
}

int main() {
  relaxed_add_kernel_test();
  relaxed_or_kernel_test();
  relaxed_xor_kernel_test();
  relaxed_and_kernel_test();
  relaxed_max_kernel_test();
  relaxed_min_kernel_test();

  return 0;
}
