#include <iostream>
#include <queue>
#include <cuda_runtime.h>

#define N 25000
#define BLOCK_SIZE 256

void random_adjacency_matrix(float *matrix, int n)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = rand() % 2;
        }
    }

    for (int i = 0; i < n; i++) {
        matrix[i * n + i] = 0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[j * n + i] = matrix[i * n + j];
        }
    }
}

__global__ void bfs(float *dev_graph, int *dev_visited, int numVertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices && dev_visited[tid] == 0) {
        extern __shared__ int shared_visited[];
        shared_visited[threadIdx.x] = dev_visited[tid];
        __syncthreads();

        for (int level = 0; level < numVertices; level++) {
            if (shared_visited[threadIdx.x] == 0 && dev_graph[tid * numVertices + level] == 1) {
                shared_visited[threadIdx.x] = 1;
                break;
            }
        }

        __syncthreads();
        dev_visited[tid] = shared_visited[threadIdx.x];
    }
}

void parallel_bfs(float *dev_graph, int index)
{
    int *dev_visited;
    cudaMallocManaged((void **)&dev_visited, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        dev_visited[i] = 0;
    }

    dev_visited[index] = 1;

    int numVertices = N;
    int numBlocks = (numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    clock_t par_begin = clock();

    while (true) {
        bfs<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int), stream>>>(dev_graph, dev_visited, numVertices);

        bool done = true;
        cudaStreamSynchronize(stream);

        for (int i = 0; i < numVertices; i++) {
            if (dev_visited[i] == 0) {
                done = false;
                break;
            }
        }

        if (done) {
            break;
        }
    }

    clock_t par_end = clock();
    /*for(int i = 0; i < N; i++) {
        if(dev_visited[i] == 1) {
            std::cout<<i<<" ";
        }
	
    }*/
    cudaFree(dev_visited);
    cudaStreamDestroy(stream);

    double par_elapsed_secs = double(par_end - par_begin) / CLOCKS_PER_SEC;
    std::cout << "Elapsed Time for parallel BFS: " << par_elapsed_secs << "\n";
}

int main()
{
    float *dev_graph;
    cudaMallocManaged((void **)&dev_graph, N * N * sizeof(float));
    random_adjacency_matrix(dev_graph, N);
    parallel_bfs(dev_graph, 0);

    cudaFree(dev_graph);

    return 0;
}
