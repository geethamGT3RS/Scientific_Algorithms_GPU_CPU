#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#define WIDTH 500
#define HEIGHT 500
#define RADIUS 200
#define CENTER_X WIDTH / 2
#define CENTER_Y HEIGHT / 2

void renderCircle(unsigned char* image) {
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
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
    }
}

int main() {
    // Allocate memory for the image
    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    // Initialize the image to black (0)
    memset(image, 0, WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    // Render the circle
    renderCircle(image);

    // Save the image as a PPM file
    FILE* file = fopen("circle.ppm", "wb");
    fprintf(file, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, sizeof(unsigned char), WIDTH * HEIGHT * 3, file);
    fclose(file);

    // Free memory
    free(image);

    return 0;
}

