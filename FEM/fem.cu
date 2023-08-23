#include <iostream>
#include <vector>
#include <cuda_runtime.h>


__global__ void calculateTemperature(double *theta, double *T, int m, int n, double dx, double dt, int Nbi, double phi, double b, double k, double T0, double T1)
{
    int i = (m-2)-(blockIdx.y * blockDim.y + threadIdx.y);
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >=0 && j < n-1)
    {
        // BC
        if (i ==0)
        {
            theta[i * n + j+1] = theta[(i + 1) * n + j+1] * ((1 + (dx / 4) * Nbi) / (1 - (dx / 4) * Nbi));
			T[i * n + j]= T0 + theta[i * n + j] * (T1 - T0);
        }
      
        else
        {
            double prevTheta = theta[i * n + j];
            theta[i * n + j+1] = theta[i*n+j] + (dt / (dx * dx)) * (theta[(i + 1) * n + j] - 2 * prevTheta + theta[(i - 1) * n + j]) +
                                dt * (phi * (b * b)) / (k * (T1 - T0));
			T[i * n + j]= T0 + theta[i * n + j] * (T1 - T0);
        }
		
     }
		
		
}

int main()
{
    double phi = 4e8;
    double b = 0.01;
    
    double k = 373;
    double T0 = 30;
    double T1 = 100;

    int m = 500;
    int n = 500;

    double dx = 0.2;
    double dt = 0.02;

    int Nbi = 10;

    std::vector<double> theta(m * n, 0.0);
    std::vector<double> T(m * n, 0.0);

		
		for(int i = 0 ; i < m ; i++)
			for(int j = 0 ; j <n ; j ++)
		{
		 if (j == 0)
        {
            theta[i * n + j] = 0.0;
        }
        else if (i == m - 1)
        {
            theta[i * n + j] = 1.0;
        }
		}
    // Device memory allocation
    double *d_theta, *d_T;
    cudaMalloc((void **)&d_theta, m * n * sizeof(double));
    cudaMalloc((void **)&d_T, m * n * sizeof(double));

    // Copy data to device
	
    cudaMemcpy(d_theta, theta.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Kernel invocation
	clock_t par_begin = clock();
	for(int i =0 ; i< n-1 ; i++)
    calculateTemperature<<<gridSize, blockSize>>>(d_theta, d_T, m, n, dx, dt, Nbi, phi, b, k, T0, T1);
	clock_t par_end = clock();
    // Copy results back to host
    cudaMemcpy(T.data(), d_T, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_theta);
    cudaFree(d_T);

    // Print temperature matrix
	for(int j = 0 ; j < n-1 ; j++)
	{
		T[(m-1)*n+j] = 100;
	}
	
    /*for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n-1; j++)
        {
            std::cout << T[i * n + j] << " ";
        }
        std::cout << std::endl;
    }*/
	double par_elapsed_secs = double(par_end - par_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for parallel FEM: "  <<par_elapsed_secs<<"\n";

    return 0;
}
