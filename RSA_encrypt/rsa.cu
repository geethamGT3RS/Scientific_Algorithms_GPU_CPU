#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

__device__ int modularExponentiation(int base, unsigned int exponent, int modulus) {
    int result = 1;
    base = base % modulus;
    while (exponent > 0) {
        if (exponent % 2 == 1)
            result = (result * base) % modulus;
        exponent = exponent >> 1;
        base = (base * base) % modulus;
    }
    return result;
}

__global__ void rsaEncrypt(int *message, int publicKey, int modulus, int *encryptedMessage, int messageSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < messageSize; i += stride) {
        encryptedMessage[i] = modularExponentiation(message[i], publicKey, modulus);
    }
}
int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}
int main() {
    int p=3, q=7;
    int publicKey, modulus;
    int messageSize=50000000;
    int *message, *encryptedMessage;
    int *devMessage, *devEncryptedMessage;
    int phi = (p - 1) * (q - 1);
    publicKey = 2;
    while (publicKey < phi) {
        if (gcd(publicKey, phi) == 1)
            break;
        publicKey++;
    }
    modulus = p * q;
    message = (int*)malloc(messageSize * sizeof(int));
    encryptedMessage = (int*)malloc(messageSize * sizeof(int));

    printf("Enter the message: ");
    for (int i = 0; i < messageSize; i++) {
        message[i]=rand();
    }

    cudaMalloc((void**)&devMessage, messageSize * sizeof(int));
    cudaMalloc((void**)&devEncryptedMessage, messageSize * sizeof(int));
    cudaMemcpy(devMessage, message, messageSize * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (messageSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rsaEncrypt<<<numBlocks, THREADS_PER_BLOCK>>>(devMessage, publicKey, modulus, devEncryptedMessage, messageSize);

    cudaMemcpy(encryptedMessage, devEncryptedMessage, messageSize * sizeof(int), cudaMemcpyDeviceToHost);
    free(message);
    free(encryptedMessage);
    cudaFree(devMessage);
    cudaFree(devEncryptedMessage);

    return 0;
}
/*
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256

// RSA Encryption parameters
#define RSA_N 187
#define RSA_E 7

// Device function for modular exponentiation
__device__ int modExp(int base, int exp, int modulus) {
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp = exp / 2;
    }
    return result;
}

// Kernel function for RSA encryption
__global__ void rsaEncrypt(int* plaintext, int* ciphertext, int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wordsPerThread = numElements / (blockDim.x * gridDim.x);
    int start = tid * wordsPerThread;
    int end = start + wordsPerThread;

    for (int i = start; i < end; i++) {
        int word = plaintext[i];
        ciphertext[i] = modExp(word, RSA_E, RSA_N);
    }
}

int main() {
    // Input plaintext
    int plaintext[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int numElements = sizeof(plaintext) / sizeof(int);
    int* d_plaintext;
    int* d_ciphertext;

    // Allocate device memory
    cudaMalloc((void**)&d_plaintext, sizeof(int) * numElements);
    cudaMalloc((void**)&d_ciphertext, sizeof(int) * numElements);

    // Copy input data from host to device
    cudaMemcpy(d_plaintext, plaintext, sizeof(int) * numElements, cudaMemcpyHostToDevice);

    // Launch the kernel
    int numBlocks = (numElements + NUM_THREADS - 1) / NUM_THREADS;
    rsaEncrypt<<<numBlocks, NUM_THREADS>>>(d_plaintext, d_ciphertext, numElements);

    // Copy the result back to host
    int* ciphertext = (int*)malloc(sizeof(int) * numElements);
    cudaMemcpy(ciphertext, d_ciphertext, sizeof(int) * numElements, cudaMemcpyDeviceToHost);

    // Print the encrypted ciphertext
    printf("Ciphertext: ");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", ciphertext[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);

    // Free host memory
    free(ciphertext);

    return 0;
}
*/