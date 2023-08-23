#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 500
#define HEIGHT 500
#define RADIUS 200
#define CENTER_X WIDTH / 2
#define CENTER_Y HEIGHT / 2

__global__
void renderCircle(unsigned char* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the pixel coordinates
    int x = idx;
    int y = idy;

    // Calculate the distance from the center of the circle
    float distance = sqrtf(powf(x - CENTER_X, 2) + powf(y - CENTER_Y, 2));

    // Check if the pixel is within the circle
    if (distance <= RADIUS) {
        // Set the pixel color to white (255)
        int pixelIndex = (y * WIDTH + x) * 3;
        image[pixelIndex] = 255;        // Red channel
        image[pixelIndex + 1] = 255;    // Green channel
        image[pixelIndex + 2] = 255;    // Blue channel
    }
}

int main() {
    // Allocate memory for the image
    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    // Initialize the image to black (0)
    memset(image, 0, WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    // Allocate memory on the GPU
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    // Copy the image data from host to device
    cudaMemcpy(d_image, image, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 dimGrid(WIDTH / 16, HEIGHT / 16);
    dim3 dimBlock(16, 16);

    // Call the kernel to render the circle
    renderCircle<<<dimGrid, dimBlock>>>(d_image);

    // Copy the image data from device to host
    cudaMemcpy(image, d_image, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_image);

    // Save the image as a PPM file
    FILE* file = fopen("circle.ppm", "wb");
    fprintf(file, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, sizeof(unsigned char), WIDTH * HEIGHT * 3, file);
    fclose(file);

    // Free memory on the host
    free(image);

    return 0;
}

