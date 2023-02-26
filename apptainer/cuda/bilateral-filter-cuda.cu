/* Pietro Pezzi - pietro.pezzi3@studio.unibo.it - 0000925022 */
#include "hpc.h"
#include "ppm.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.142857

#define BLKDIM 16
#define RADIUS 10

typedef struct {
    double red, green, blue;
} RGB_Double;

// Sets all parameters of given RGB_Double with the given value.
__device__ void set_dpixel(RGB_Double *pixel, double value)
{
    pixel->blue = value;
    pixel->green = value;
    pixel->red = value;
}

/* Simplifies indexing on a M*N grid. */
__device__ int IDX(int i, int j, int M, int N)
{
    /* wrap-around */
    i = (i + M) % M;
    j = (j + N) % N;
    return i * N + j;
}
__device__ double gauss_kernel(int x, double sigma)
{
    return (1 / (sqrt((double)(2.0 * PI)) * sigma)) *
           exp((double)(-(x * x) / (2.0 * (sigma * sigma))));
}

/* Returns the eucliedean distance between two points (x1,y1) and (x2,y2). */
__device__ double eucl_dist(int x1, int y1, int x2, int y2)
{
    return sqrt((double)((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)));
}

/* Kernel function to apply bilateral filtering algorithm on NxN image.
 *
 * d_imgin is the RGB_Pixel pointer to the original image in the GPU global
 * memory. 
 * d_imgout is the RGB_Pixel pointer to the RGB_Pixel array where the
 * result of the filter will be stored, in the GPU Global memory. 
 * N in image's width/height. 
 * The parameter sigma_color determines the range of values that
 * will have a considerable weight when computing the new pixel value, given
 * that the value of a pixel is \in [0, 255] sigma_color should be set
 * accordingly. 
 * The parameter sigma_spatial determines the range of distance
 * that will have a considerable weight when computing the new pixel value. The
 * parameter filtered is the RGB_Pixel pointer where the new pixel's values will
 * be stored.
 */
__global__ void bilateral_filter(RGB_Pixel *d_imgin, RGB_Pixel *d_imgout, int N,
                                 int sigma_color, int sigma_spatial)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x + RADIUS;
    const int local_y = threadIdx.y + RADIUS;
    int i, j;

    const int dim = BLKDIM + (RADIUS * 2);
    __shared__ RGB_Pixel local_array[dim * dim];

    // Load local_array.
    // load "personal" pixel.
    local_array[IDX(local_x, local_y, dim, dim)] = d_imgin[IDX(x, y, N, N)];

    // Load "sides" ghost pixels.
    if (local_x == RADIUS) {
        for (i = 1; i <= RADIUS; i++) {
            local_array[IDX(local_x - i, local_y, dim, dim)] =
                d_imgin[IDX(x - i, y, N, N)];
        }
    }
    if (local_x == RADIUS + blockDim.x - 1) {
        for (i = 1; i <= RADIUS; i++) {
            local_array[IDX(local_x + i, local_y, dim, dim)] =
                d_imgin[IDX(x + i, y, N, N)];
        }
    }
    if (local_y == RADIUS) {
        for (i = 1; i <= RADIUS; i++) {
            local_array[IDX(local_x, local_y - i, dim, dim)] =
                d_imgin[IDX(x, y - i, N, N)];
        }
    }
    if (local_y == RADIUS + blockDim.y - 1) {
        for (i = 1; i <= RADIUS; i++) {
            local_array[IDX(local_x, local_y + i, dim, dim)] =
                d_imgin[IDX(x, y + i, N, N)];
        }
    }

    // Load "corner" ghost pixels.
    if (local_x == RADIUS && local_y == RADIUS) {
        for (i = 1; i <= RADIUS; i++) {
            for (j = 1; j <= RADIUS; j++) {
                local_array[IDX(local_x - i, local_y - j, dim, dim)] =
                    d_imgin[IDX(x - i, y - j, N, N)];
            }
        }
    }
    if (local_x == RADIUS && local_y == RADIUS + BLKDIM - 1) {
        for (i = 1; i <= RADIUS; i++) {
            for (j = 1; j <= RADIUS; j++) {
                local_array[IDX(local_x - i, local_y + j, dim, dim)] =
                    d_imgin[IDX(x - i, y + j, N, N)];
            }
        }
    }
    if (local_x == RADIUS + BLKDIM - 1 && local_y == RADIUS) {
        for (i = 1; i <= RADIUS; i++) {
            for (j = 1; j <= RADIUS; j++) {
                local_array[IDX(local_x + i, local_y - j, dim, dim)] =
                    d_imgin[IDX(x + i, y - j, N, N)];
            }
        }
    }
    if (local_x == RADIUS + BLKDIM - 1 && local_y == RADIUS + BLKDIM - 1) {
        for (i = 1; i <= RADIUS; i++) {
            for (j = 1; j <= RADIUS; j++) {
                local_array[IDX(local_x + i, local_y + j, dim, dim)] =
                    d_imgin[IDX(x + i, y + j, N, N)];
            }
        }
    }
    __syncthreads();

    // Apply Bilateral filter on pixel.
    RGB_Double total_weight, partial_weight, neighborhood, gauss_color;
    double gauss_spatial;

    set_dpixel(&total_weight, 0.0);
    set_dpixel(&partial_weight, 0.0);
    set_dpixel(&neighborhood, 0.0);
    for (i = -RADIUS; i <= RADIUS; i++) {
        for (j = -RADIUS; j <= RADIUS; j++) {
            gauss_color.red = gauss_kernel(
                local_array[IDX(local_x + i, local_y + j, dim, dim)].red -
                    local_array[IDX(local_x, local_y, dim, dim)].red,
                sigma_color);
            gauss_color.green = gauss_kernel(
                local_array[IDX(local_x + i, local_y + j, dim, dim)].green -
                    local_array[IDX(local_x, local_y, dim, dim)].green,
                sigma_color);
            gauss_color.blue = gauss_kernel(
                local_array[IDX(local_x + i, local_y + j, dim, dim)].blue -
                    local_array[IDX(local_x, local_y, dim, dim)].blue,
                sigma_color);
            gauss_spatial = gauss_kernel(
                eucl_dist(local_x + i, local_y + j, local_x, local_y),
                sigma_spatial);
            partial_weight.red = gauss_color.red * gauss_spatial;
            partial_weight.green = gauss_color.green * gauss_spatial;
            partial_weight.blue = gauss_color.blue * gauss_spatial;
            neighborhood.red +=
                (local_array[IDX(local_x + i, local_y + j, dim, dim)].red *
                 partial_weight.red);
            neighborhood.green +=
                (local_array[IDX(local_x + i, local_y + j, dim, dim)].green *
                 partial_weight.green);
            neighborhood.blue +=
                (local_array[IDX(local_x + i, local_y + j, dim, dim)].blue *
                 partial_weight.blue);
            total_weight.red += partial_weight.red;
            total_weight.green += partial_weight.green;
            total_weight.blue += partial_weight.blue;
        }
    }

    // Update pixel value in global result array.
    d_imgout[IDX(x, y, N, N)].red = (int)(neighborhood.red / total_weight.red);
    d_imgout[IDX(x, y, N, N)].green =
        (int)(neighborhood.green / total_weight.green);
    d_imgout[IDX(x, y, N, N)].blue =
        (int)(neighborhood.blue / total_weight.blue);
}

int main(int argc, char *argv[])
{
    FILE *in = NULL;
    PPM_Image *img = NULL;
    double sigma_color, sigma_spatial;
    RGB_Pixel *d_imgin, *d_imgout;
    double t1, t2;

    if (argc != 4) {
        fprintf(stderr, "USAGE: <imagepath> <sigma_color> <sigma_spatial>.\n");
        return 1;
    }
    if ((in = fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "FATAL: Image file '%s' not found.\n", argv[1]);
        return 1;
    }
    if (sscanf(argv[2], "%lf", &sigma_color) != 1) {
        fprintf(stderr,
                "FATAL: Error while reading the sigma-color's value.\n");
        fclose(in);
        return 1;
    }
    if (sscanf(argv[3], "%lf", &sigma_spatial) != 1) {
        fprintf(stderr,
                "FATAL: Error while reading the sigma-spatial's value.\n");
        fclose(in);
        return 1;
    }

    img = (PPM_Image *)malloc(sizeof(PPM_Image));
    assert(img != NULL);
    if (load_PPM(in, img) == 1) {
        fprintf(stderr, "FATAL: Error while loading image.\n");
        fclose(in);
        return 1;
    }
    fclose(in);

    if (img->header.height != img->header.width) {
        fprintf(stderr, "FATAL: Image's width and height must be equal.\n");
        free(img);
        return 1;
    }
    if (img->header.height % BLKDIM != 0) {
        fprintf(stderr,
                "FATAL: Image's height and width must be a multiple of %d.\n",
                BLKDIM);
        free(img);
        return 1;
    }

    const dim3 block(BLKDIM, BLKDIM);
    const dim3 grid((img->header.height + BLKDIM - 1) / BLKDIM,
                    (img->header.height + BLKDIM - 1) / BLKDIM);

    const size_t img_size =
        (img->header.height * img->header.width) * sizeof(RGB_Pixel);
    cudaSafeCall(cudaMalloc((void **)&d_imgin, img_size));
    cudaSafeCall(cudaMalloc((void **)&d_imgout, img_size));
    cudaSafeCall(
        cudaMemcpy(d_imgin, img->pixelArray, img_size, cudaMemcpyHostToDevice));

    t1 = hpc_gettime();
    bilateral_filter<<<grid, block>>>(d_imgin, d_imgout, img->header.width,
                                      sigma_color, sigma_spatial);
    cudaCheckError();
    cudaDeviceSynchronize();
    t2 = hpc_gettime();

    cudaSafeCall(cudaMemcpy(img->pixelArray, d_imgout, img_size,
                            cudaMemcpyDeviceToHost));

    if (write_PPM("filtered_image_cuda.ppm", img) == 1) {
        cudaFree(d_imgin);
        cudaFree(d_imgout);
        free(img);
        return 1;
    }

    printf("Cuda imp execution time: %f\n", t2-t1);
    cudaFree(d_imgin);
    cudaFree(d_imgout);
    free(img);
    return 0;
}
