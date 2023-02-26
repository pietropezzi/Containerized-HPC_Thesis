/* Pietro Pezzi - pietro.pezzi3@studio.unibo.it - 0000925022 */
#include "ppm.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.142857

typedef struct {
    double red, green, blue;
} RGB_Double;

// Sets all parameters of given RGB_Double with the given value.
void set_dpixel(RGB_Double *pixel, double value)
{
    pixel->blue = value;
    pixel->green = value;
    pixel->red = value;
}

/* Simplifies indexing on a M*N grid. */
int IDX(int i, int j, int M, int N)
{
    /* wrap-around */
    i = (i + M) % M;
    j = (j + N) % N;
    return i * N + j;
}

/* Returns the gaussian kernel's calculated in x and with given sigma. */
double gauss_kernel(int x, double sigma)
{
    return (1 / (sqrt(2.0 * PI) * sigma)) *
           exp(-(x * x) / (2.0 * (sigma * sigma)));
}

/* Returns the eucliedean distance between two points (x1,y1) and (x2,y2). */
double eucl_dist(int x1, int y1, int x2, int y2)
{
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/*
 * Given a PPM_Image pointer, the bilateral_filter() function applyes the
 * bilateral filtering algorithm on the given image.
 * The parameter radius represents the radius of the square that will be
 * used as each pixel's neighborhood size.
 *   _ _ _ _ _
 *  |    r    |   with rad=2 the bilateral_filter will use a (5x5) neighborhood.
 *  |    r    |   So with a given radius r the function will use a
 *  |r r P    |   ((2*r)+1 x (2*r)+1) neighborhood, since the central
 *  |         |   pixel is added.
 *  |_ _ _ _ _|
 *
 * The parameter sigma_color determines the range of values that will have a
 * considerable weight when computing the new pixel value, given that the value
 * of a pixel is \in [0, 255] sigma_color should be set accordingly.
 * The parameter sigma_spatial determines the range of distance that will have a
 * considerable weight when computing the new pixel value.
 * The parameter filtered is the RGB_Pixel pointer where the new pixel's
 * values will be stored.
 */
void bilateral_filter(PPM_Image *img, RGB_Pixel *filtered, int radius,
                      double sigma_color, double sigma_spatial)
{
    int i, j, k, l;
    RGB_Double total_weight, partial_weight, neighborhood, gauss_color;
    double gauss_spatial;
    const int M = img->header.height;
    const int N = img->header.width;
    for (i = 0; i <= M; i++) {
        for (j = 0; j <= N; j++) {
            set_dpixel(&total_weight, 0.0);
            set_dpixel(&partial_weight, 0.0);
            set_dpixel(&neighborhood, 0.0);
            for (k = -radius; k <= radius; k++) {
                for (l = -radius; l <= radius; l++) {
                    gauss_color.red = gauss_kernel(
                        img->pixelArray[IDX(i + k, j + l, M, N)].red -
                            img->pixelArray[IDX(i, j, M, N)].red,
                        sigma_color);
                    gauss_color.green = gauss_kernel(
                        img->pixelArray[IDX(i + k, j + l, M, N)].green -
                            img->pixelArray[IDX(i, j, M, N)].green,
                        sigma_color);
                    gauss_color.blue = gauss_kernel(
                        img->pixelArray[IDX(i + k, j + l, M, N)].blue -
                            img->pixelArray[IDX(i, j, M, N)].blue,
                        sigma_color);
                    gauss_spatial = gauss_kernel(eucl_dist(i + k, j + l, i, j),
                                                 sigma_spatial);
                    partial_weight.red = gauss_color.red * gauss_spatial;
                    partial_weight.green = gauss_color.green * gauss_spatial;
                    partial_weight.blue = gauss_color.blue * gauss_spatial;
                    neighborhood.red +=
                        (img->pixelArray[IDX(i + k, j + l, M, N)].red *
                         partial_weight.red);
                    neighborhood.green +=
                        (img->pixelArray[IDX(i + k, j + l, M, N)].green *
                         partial_weight.green);
                    neighborhood.blue +=
                        (img->pixelArray[IDX(i + k, j + l, M, N)].blue *
                         partial_weight.blue);
                    total_weight.red += partial_weight.red;
                    total_weight.green += partial_weight.green;
                    total_weight.blue += partial_weight.blue;
                }
            }
            filtered[IDX(i, j, M, N)].red =
                (int)(neighborhood.red / total_weight.red);
            filtered[IDX(i, j, M, N)].green =
                (int)(neighborhood.green / total_weight.green);
            filtered[IDX(i, j, M, N)].blue =
                (int)(neighborhood.blue / total_weight.blue);
        }
    }
}

int main(int argc, char *argv[])
{
    FILE *in = NULL;
    PPM_Image *img = NULL;
    int radius;
    double sigma_color, sigma_spatial;

    if (argc != 5) {
        fprintf(stderr,
                "USAGE: <imagepath> <radius> <sigma_color> <sigma_spatial>.\n");
        return 1;
    }
    if ((in = fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "FATAL: Image file '%s' not found.\n", argv[1]);
        return 1;
    }
    if (sscanf(argv[2], "%d", &radius) != 1) {
        fprintf(stderr, "FATAL: Error while reading the radius's value.\n");
        fclose(in);
        return 1;
    }
    if (sscanf(argv[3], "%lf", &sigma_color) != 1) {
        fprintf(stderr,
                "FATAL: Error while reading the sigma-color's value.\n");
        fclose(in);
        return 1;
    }
    if (sscanf(argv[4], "%lf", &sigma_spatial) != 1) {
        fprintf(stderr,
                "FATAL: Error while reading the sigma-spatial's value.\n");
        fclose(in);
        return 1;
    }

    img = malloc(sizeof(PPM_Image));
    assert(img != NULL);
    if (load_PPM(in, img) == 1) {
        fprintf(stderr, "FATAL: Error while loading image.\n");
        fclose(in);
        return 1;
    }
    fclose(in);

    const size_t img_size =
        (img->header.width * img->header.height) * sizeof(RGB_Pixel);
    RGB_Pixel *filtered = malloc(img_size * sizeof(*filtered));
    assert(filtered != NULL);

    bilateral_filter(img, filtered, radius, sigma_color, sigma_spatial);
    img->pixelArray = filtered;

    if (write_PPM("filtered_image.ppm", img) == 1) {
        free(filtered);
        free(img);
        return 1;
    }
    free(filtered);
    free(img);
    return 0;
}
