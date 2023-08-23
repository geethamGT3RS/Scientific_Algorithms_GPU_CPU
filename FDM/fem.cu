#include <iostream>
#include <cmath>
#include <vector>
#include<cuda_runtime.h>

__global__ void calculateTheta(double* theta, double* theta_guess, double dx, double dy, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < m - 1 && j < n - 1) {
        theta[i * n + j] = (1.0 / (2.0 * (1.0 + pow(dx / dy, 2)))) *
                           (theta_guess[(i + 1) * n + j] + theta_guess[(i - 1) * n + j] +
                            pow(dx / dy, 2) * (theta_guess[i * n + (j + 1)] + theta_guess[i * n + (j - 1)]));
    }
}

__global__ void updateThetaGuess(double* theta_guess, double* theta , double* theta_new_guess, double w, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        theta_new_guess[i * n + j] = theta_guess[i * n + j] + w * (theta[i * n + j] - theta_guess[i * n + j]);
    }
}

int main() {
    
        int m = 4096; // x axis
        int n = 4096; // y axis

        double b = 0.5;
        double L = 1;

        double beta = L / b;

        double dx = 1.0 / (m - 1);
        double dy = beta / (n - 1);

        double w = 0.75;
        std::vector<double> theta(m * n, 0.0);
        std::vector<double> theta_guess(m * n, 3.0 / 7.0);

        for (int i = 0; i < m; i++) {
            theta[i * n] = 0.0;
            theta[i * n + n - 1] = 1.0;
            theta_guess[i * n] = 0.0;
            theta_guess[i * n + n - 1] = 1.0;
        }

        std::vector<double> theta_new_guess(m * n, 0.0);
        double sum = 0.0;
        int iterations = 0;
//        double tolerance = 0.1;

        double* d_theta;
        double* d_theta_guess;
        double* d_theta_new_guess;

        cudaMalloc((void**)&d_theta, sizeof(double) * m * n);
        cudaMalloc((void**)&d_theta_guess, sizeof(double) * m * n);
        cudaMalloc((void**)&d_theta_new_guess, sizeof(double) * m * n);

       // while (sum > tolerance || iterations <= 2) {
	while(iterations<=20){
            iterations++;
            sum = 0.0;

            cudaMemcpy(d_theta, theta.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_theta_guess, theta_guess.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice);

            dim3 blockSize(32, 32);
            dim3 gridSize((n - 2 + blockSize.x - 1) / blockSize.x, (m - 2 + blockSize.y - 1) / blockSize.y);
            calculateTheta<<<gridSize, blockSize>>>(d_theta, d_theta_guess, dx, dy, m, n);

            cudaMemcpy(theta.data(), d_theta, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    sum += std::abs(theta[i * n + j] - theta_guess[i * n + j]);
                }
            }

            cudaMemcpy(d_theta_new_guess, theta_new_guess.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice);
            updateThetaGuess<<<gridSize, blockSize>>>(d_theta_guess,d_theta, d_theta_new_guess, w, m, n);
            cudaMemcpy(theta_guess.data(), d_theta_new_guess, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
        }

        std::vector<std::vector<double>> T(m, std::vector<double>(n, 0.0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T[i][j] = theta[i * n + j] * 70 + 30;
            }
        }

        // Print the result
       /* for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << T[i][j] << " ";
            }
            std::cout << std::endl;
        }*/

        cudaFree(d_theta);
        cudaFree(d_theta_guess);
        cudaFree(d_theta_new_guess);
    

    return 0;
}
