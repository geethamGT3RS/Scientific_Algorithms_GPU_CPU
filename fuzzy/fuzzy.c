#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define MAX_ITERATIONS 2
#define EPSILON 0.0001

void fuzzyCMeansClustering(float* data, int numDataPoints, int numFeatures, int numClusters, float fuzziness, float** centroids) {
    // Allocate memory for membership matrix
    float** membership = (float**)malloc(numDataPoints * sizeof(float*));
    for (int i = 0; i < numDataPoints; ++i) {
        membership[i] = (float*)malloc(numClusters * sizeof(float));
    }
    
    // Initialize membership matrix randomly
    for (int i = 0; i < numDataPoints; ++i) {
        float sum = 0.0;
        for (int j = 0; j < numClusters; ++j) {
            membership[i][j] = (float)rand() / RAND_MAX;
            sum += membership[i][j];
        }
        // Normalize membership values
        for (int j = 0; j < numClusters; ++j) {
            membership[i][j] /= sum;
        }
    }
    
    // Allocate memory for centroid matrix
    *centroids = (float*)malloc(numClusters * numFeatures * sizeof(float));
    
    // Initialize centroids randomly
    for (int i = 0; i < numClusters; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            (*centroids)[i * numFeatures + j] = (float)rand() / RAND_MAX;
        }
    }
    
    // Iterate until convergence or maximum iterations reached
    int iteration = 0;
    float diff;

    while (iteration < MAX_ITERATIONS && diff > EPSILON){
        // Create a copy of the membership matrix for comparison
        float** membershipOld = (float**)malloc(numDataPoints * sizeof(float*));
        for (int i = 0; i < numDataPoints; ++i) {
            membershipOld[i] = (float*)malloc(numClusters * sizeof(float));
            for (int j = 0; j < numClusters; ++j) {
                membershipOld[i][j] = membership[i][j];
            }
        }

        // Update membership values
        for (int i = 0; i < numDataPoints; ++i) {
            for (int j = 0; j < numClusters; ++j) {
                float sum = 0.0;
                for (int k = 0; k < numClusters; ++k) {
                    float distance_ij = 0.0;
                    for (int f = 0; f < numFeatures; ++f) {
                        float diff = data[i * numFeatures + f] - (*centroids)[j * numFeatures + f];
                        distance_ij += diff * diff;
                    }
                    float ratio = sqrt(distance_ij / (distance_ij + EPSILON));
                    sum += powf(ratio, 2.0 / (fuzziness - 1.0));
                }
                membership[i][j] = 1.0 / sum;
            }
        }
        
        // Update centroids
        for (int j = 0; j < numClusters; ++j) {
            float sum = 0.0;
            for (int i = 0; i < numDataPoints; ++i) {
                float membershipPower = powf(membership[i][j], fuzziness);
                for (int f = 0; f < numFeatures; ++f) {
                    (*centroids)[j * numFeatures + f] += membershipPower * data[i * numFeatures + f];
                }
                sum += membershipPower;
            }
            for (int f = 0; f < numFeatures; ++f) {
                (*centroids)[j * numFeatures + f] /= sum;
            }
        }
        
        // Calculate maximum difference between old and new membership values
        diff = 0.0;
        for (int i = 0; i < numDataPoints; ++i) {
            for (int j = 0; j < numClusters; ++j) {
                diff = fmaxf(diff, fabsf(membership[i][j] - membershipOld[i][j]));
            }
        }

        // Free memory for membershipOld
        for (int i = 0; i < numDataPoints; ++i) {
            free(membershipOld[i]);
        }
        free(membershipOld);
        
        iteration++;
    } while (iteration < MAX_ITERATIONS && diff > EPSILON);
    
    // Free memory for membership matrix
    for (int i = 0; i < numDataPoints; ++i) {
        free(membership[i]);
    }
    free(membership);
}

int main() {
    // Example usage
    int numDataPoints= 50;
    int numFeatures = 5;
    int numClusters = 5;
    float fuzziness = 20.0;

    // Generate random data
    float* data = (float*)malloc(numDataPoints * numFeatures * sizeof(float));
    for (int i = 0; i < numDataPoints; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            data[i * numFeatures + j] = (float)rand() / RAND_MAX;
        }
    }
    
    // Perform fuzzy clustering
    float* centroids;
    fuzzyCMeansClustering(data, numDataPoints, numFeatures, numClusters, fuzziness, &centroids);
    
    // Print centroids
    printf("Centroids:\n");
    for (int i = 0; i < numClusters; ++i) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < numFeatures; ++j) {
            printf("%.6f ", centroids[i * numFeatures + j]);
        }
        printf("\n");
    }

    // Free memory
    free(data);
    free(centroids);
    
    return 0;
}

