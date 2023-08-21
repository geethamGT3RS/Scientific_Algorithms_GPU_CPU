#include <cuda_runtime.h>
#include <stdio.h>

__global__ void membershipCalculationKernel(const float* data, const float* centroids, float* memberships, int numDataPoints,
                                            int numClusters, int numFeatures,
                                            float fuzziness) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numDataPoints) {
        for (int k = 0; k < numClusters; ++k) {
            float numerator = 0.0;
            float denominator = 0.0;
            for (int j = 0; j < numFeatures; ++j) {
                numerator += powf(data[i * numFeatures + j] - centroids[k * numFeatures + j], 2.0 / (fuzziness - 1));
            }
            for (int l = 0; l < numClusters; ++l) {
                float temp = 0.0;
                for (int j = 0; j < numFeatures; ++j) {
                    temp += powf(data[i * numFeatures + j] - centroids[l * numFeatures + j], 2.0 / (fuzziness - 1));
                }
                denominator += powf(numerator / temp, fuzziness);
            }
            memberships[i * numClusters + k] = 1.0 / denominator;
        }
    }
}

__global__ void centroidUpdateKernel(const float* data, const float* memberships,
                                     float* centroids, int numDataPoints,
                                     int numClusters, int numFeatures,
                                     float fuzziness) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < numClusters) {
        for (int j = 0; j < numFeatures; ++j) {
            float numerator = 0.0;
            float denominator = 0.0;
            for (int i = 0; i < numDataPoints; ++i) {
                float membershipPow = powf(memberships[i * numClusters + k], fuzziness);
                numerator += membershipPow * data[i * numFeatures + j];
                denominator += membershipPow;
            }
            centroids[k * numFeatures + j] = numerator / denominator;
        }
    }
}

void fuzzyCMeansClustering(const float* data, int numDataPoints, int numFeatures,
                           int numClusters, float fuzziness, int maxIterations,
                           float* centroids) {
    // Allocate device memory
    float* dataDevice;
    float* centroidsDevice;
    float* membershipsDevice;

    cudaMalloc((void**)&dataDevice, numDataPoints * numFeatures * sizeof(float));
    cudaMalloc((void**)&centroidsDevice, numClusters * numFeatures * sizeof(float));
    cudaMalloc((void**)&membershipsDevice, numDataPoints * numClusters * sizeof(float));

    // Copy data and centroids to device memory
    cudaMemcpy(dataDevice, data, numDataPoints * numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(centroidsDevice, centroids, numClusters * numFeatures * sizeof(float), cudaMemcpyHostToDevice);

    // Set the number of threads per block and calculate the number of blocks
    int threadsPerBlock = 256;
    int numBlocks = (numDataPoints + threadsPerBlock - 1) / threadsPerBlock;

    // Perform iterations of the FCM algorithm
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Calculate memberships
        membershipCalculationKernel<<<numBlocks, threadsPerBlock>>>(dataDevice, centroidsDevice,
                                                                    membershipsDevice, numDataPoints,
                                                                    numClusters, numFeatures,
                                                                    fuzziness);

        // Synchronize device before centroid update
        cudaDeviceSynchronize();

        // Update centroids
        centroidUpdateKernel<<<numBlocks, threadsPerBlock>>>(dataDevice, membershipsDevice,
                                                             centroidsDevice, numDataPoints,
                                                             numClusters, numFeatures,
                                                             fuzziness);

        // Synchronize device after centroid update
        cudaDeviceSynchronize();
    }

    // Copy centroids from device to host memory
    cudaMemcpy(centroids, centroidsDevice, numClusters * numFeatures * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dataDevice);
    cudaFree(centroidsDevice);
    cudaFree(membershipsDevice);
}

int main() {
    // Example usage
    const int numDataPoints = 50000;
    const int numFeatures = 10;
    const int numClusters = 10;
    const float fuzziness = 2.0;
    const int maxIterations = 2;
    
    // Generate random data
    float* data = (float*)malloc(numDataPoints * numFeatures * sizeof(float));
    for (int i = 0; i < numDataPoints * numFeatures; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate memory for centroids
    float* centroids = (float*)malloc(numClusters * numFeatures * sizeof(float));
    
    // Perform fuzzy c-means clustering
    fuzzyCMeansClustering(data, numDataPoints, numFeatures, numClusters, fuzziness, maxIterations, centroids);
    
    // Print centroids
    printf("Centroids:\n");
    for (int i = 0; i < numClusters; ++i) {
        printf("Centroid %d: ", i + 1);
        for (int j = 0; j < numFeatures; ++j) {
            printf("%f ", centroids[i * numFeatures + j]);
        }
        printf("\n");
    }
    
    // Free memory
    free(data);
    free(centroids);

    return 0;
}

