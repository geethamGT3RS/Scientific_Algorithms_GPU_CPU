#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define DATASET_SIZE 10000000

// CUDA kernel to rotate a 3D vector using Euler angles
__global__ void rotateVector(float eulerX, float eulerY, float eulerZ, float* inputVectors, float* outputVectors,int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx<num){
    // Convert Euler angles to radians
    float phi = eulerX * M_PI / 180.0;
    float theta = eulerY * M_PI / 180.0;
    float psi = eulerZ * M_PI / 180.0;

    // Calculate sine and cosine values
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float sinPsi = sin(psi);
    float cosPsi = cos(psi);

    float* inputVector = &inputVectors[idx * 3];
    float* outputVector = &outputVectors[idx * 3];

    

    // Perform rotation using Euler angles
    outputVector[0] = cosTheta * cosPsi * inputVector[0] +
                      (cosPhi * sinPsi + sinPhi * sinTheta * cosPsi) * inputVector[1] +
                      (sinPhi * sinPsi - cosPhi * sinTheta * cosPsi) * inputVector[2];

    outputVector[1] = -cosTheta * sinPsi * inputVector[0] +
                      (cosPhi * cosPsi - sinPhi * sinTheta * sinPsi) * inputVector[1] +
                      (sinPhi * cosPsi + cosPhi * sinTheta * sinPsi) * inputVector[2];

    outputVector[2] = sinTheta * inputVector[0] -
                      sinPhi * cosTheta * inputVector[1] +
                      cosPhi * cosTheta * inputVector[2];
}
}

int main() {
    // Allocate memory for the dataset of input vectors on the host
    float* inputVectorsHost = (float*)malloc(DATASET_SIZE * 3 * sizeof(float));
    float* outputVectorsHost = (float*)malloc(DATASET_SIZE * 3 * sizeof(float));

    // Generate random input vectors on the host
    for (int i = 0; i < DATASET_SIZE; i++) {
        inputVectorsHost[i * 3] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // Random value between -1 and 1
        inputVectorsHost[i * 3 + 1] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
        inputVectorsHost[i * 3 + 2] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Euler angles (in degrees)
    float eulerX = 30.0;
    float eulerY = 45.0;
    float eulerZ = 60.0;

    // Allocate memory for the dataset of input vectors on the device
    float* inputVectorsDevice;
    float* outputVectorsDevice;
    cudaMalloc((void**)&inputVectorsDevice, DATASET_SIZE * 3 * sizeof(float));
    cudaMalloc((void**)&outputVectorsDevice, DATASET_SIZE * 3 * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(inputVectorsDevice, inputVectorsHost, DATASET_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (DATASET_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Start measuring execution time
    clock_t start = clock();

    // Launch the CUDA kernel to rotate the input vectors
    rotateVector<<<blocksPerGrid, threadsPerBlock>>>(eulerX, eulerY, eulerZ, inputVectorsDevice, outputVectorsDevice,DATASET_SIZE);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Stop measuring execution time
    clock_t end = clock();

    // Copy output vectors from device to host
    cudaMemcpy(outputVectorsHost, outputVectorsDevice, DATASET_SIZE * 3 * sizeof(float), cudaMemcpyDeviceToHost);
/*
   for(int i=0;i<DATASET_SIZE;i++){
	printf("Input:(%f,%f,%f) --> Output : (%f,%f,%f)\n",inputVectorsHost[i*3],inputVectorsHost[i*3 +1],inputVectorsHost[i*3 +2],outputVectorsHost[i*3],outputVectorsHost[i*3 +1],outputVectorsHost[i*3 +2]);
}
*/
    // Calculate the elapsed time in seconds
    double executionTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Display the execution time
    printf("Execution Time: %.6f seconds\n", executionTime);

    // Free memory for the dataset of input vectors on the device
    cudaFree(inputVectorsDevice);
    cudaFree(outputVectorsDevice);

    // Free memory for the dataset of input vectors on the host
    free(inputVectorsHost);
    free(outputVectorsHost);

    return 0;
}

