// ===--- z2z_2d_outofplace.cu -------------------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "cufft.h"
#include "cufftXt.h"
#include "common.h"
#include <cstring>
#include <iostream>


bool z2z_2d_outofplace() {
  cufftHandle plan_fwd;
  double2 forward_idata_h[2][5];
  set_value((double*)forward_idata_h, 20);

  double2* forward_idata_d;
  double2* forward_odata_d;
  double2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double2) * 10);
  cudaMalloc(&forward_odata_d, sizeof(double2) * 10);
  cudaMalloc(&backward_odata_d, sizeof(double2) * 10);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double2) * 10, cudaMemcpyHostToDevice);

  cufftPlan2d(&plan_fwd, 2, 5, CUFFT_Z2Z);
  cufftExecZ2Z(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[10];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 10, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[10];
  forward_odata_ref[0] =  double2{90,100};
  forward_odata_ref[1] =  double2{-23.7638,3.76382};
  forward_odata_ref[2] =  double2{-13.2492,-6.7508};
  forward_odata_ref[3] =  double2{-6.7508,-13.2492};
  forward_odata_ref[4] =  double2{3.76382,-23.7638};
  forward_odata_ref[5] =  double2{-50,-50};
  forward_odata_ref[6] =  double2{0,0};
  forward_odata_ref[7] =  double2{0,0};
  forward_odata_ref[8] =  double2{0,0};
  forward_odata_ref[9] =  double2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 10)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 10);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 10);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlan2d(&plan_bwd, 2, 5, CUFFT_Z2Z);
  cufftExecZ2Z(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double2 backward_odata_h[10];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double2) * 10, cudaMemcpyDeviceToHost);

  double2 backward_odata_ref[10];
  backward_odata_ref[0] =  double2{0,10};
  backward_odata_ref[1] =  double2{20,30};
  backward_odata_ref[2] =  double2{40,50};
  backward_odata_ref[3] =  double2{60,70};
  backward_odata_ref[4] =  double2{80,90};
  backward_odata_ref[5] =  double2{100,110};
  backward_odata_ref[6] =  double2{120,130};
  backward_odata_ref[7] =  double2{140,150};
  backward_odata_ref[8] =  double2{160,170};
  backward_odata_ref[9] =  double2{180,190};

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 10)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 10);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 10);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC z2z_2d_outofplace
  bool res = FUNC();
  cudaDeviceSynchronize();
  if (!res) {
    std::cout << "Fail" << std::endl;
    return -1;
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
#endif

