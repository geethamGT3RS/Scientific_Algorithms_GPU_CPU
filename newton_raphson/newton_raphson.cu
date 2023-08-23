#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float f(float x) {
    // Define the function for which to find the root
    return x * x - 4.0;
}

__device__ float fPrime(float x) {
    // Define the derivative of the function
    return 2.0 * x;
}

__global__ void newtonRaphsonKernel(float* result, float x0, int maxIterations) {
    float x = x0;
    for (int i = 0; i < maxIterations; ++i) {
        float delta = f(x) / fPrime(x);
        x -= delta;
        if (fabsf(delta) < 1e-6) {
            break;
        }
    }
    *result = x;
}

void newtonRaphson(float* result, float x0, int maxIterations) {
    float* resultDevice;
    cudaMalloc((void**)&resultDevice, sizeof(float));

    newtonRaphsonKernel<<<1, 1>>>(resultDevice, x0, maxIterations);

    cudaMemcpy(result, resultDevice, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(resultDevice);
}

int main() {
    float result;
    float x0 = 2.0; // Initial guess
    int maxIterations = 100;

    newtonRaphson(&result, x0, maxIterations);

    printf("Approximate root: %.6f\n", result);

    return 0;
}

