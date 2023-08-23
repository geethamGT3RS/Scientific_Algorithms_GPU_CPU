#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h>

#define DATASET_SIZE 10000000

// Function to rotate a 3D vector using Euler angles
void rotateVector(float eulerX, float eulerY, float eulerZ, float inputVector[3], float outputVector[3]) {
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

int main() {
    // Allocate memory for the dataset of input vectors
    float** inputVectors = (float**)malloc(DATASET_SIZE * sizeof(float*));
    for (int i = 0; i < DATASET_SIZE; i++) {
        inputVectors[i] = (float*)malloc(3 * sizeof(float));
    }

    // Generate random input vectors
    for (int i = 0; i < DATASET_SIZE; i++) {
        inputVectors[i][0] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // Random value between -1 and 1
        inputVectors[i][1] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
        inputVectors[i][2] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Euler angles (in degrees)
    float eulerX = 30.0;
    float eulerY = 45.0;
    float eulerZ = 60.0;



    // Apply rotation to the dataset of input vectors
    for (int i = 0; i < DATASET_SIZE; i++) {
        float outputVector[3];
        rotateVector(eulerX, eulerY, eulerZ, inputVectors[i], outputVector);
    }
       /* // Uncomment the following line to print the rotated vectors
         printf("Input Vector: (%f, %f, %f) -> Output Vector: (%f, %f, %f)\n", inputVectors[i][0], inputVectors[i][1], inputVectors[i][2], outputVector[0], outputVector[1], outputVector[2]);
   */ 
/*
    // Display the execution time
    printf("Execution Time: %.6f seconds\n", executionTime);
    // Free memory for the dataset of input vectors
    for (int i = 0; i < DATASET_SIZE; i++) {
        free(inputVectors[i]);
    }*/
    free(inputVectors);

    return 0;
}


