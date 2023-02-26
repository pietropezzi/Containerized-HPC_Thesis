/* Pietro Pezzi - pietro.pezzi3@studio.unibo.it - 0000925022 */
#include "ppm.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.142857

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* Returns a random number between 0 and 1. */
double random(void) { return (double)rand() / (double)RAND_MAX; }

/*
 * Given a PPM_Image pointer, the gaussian() function applyes Gaussian noise
 * on the given image by computing the noise of each pixel with the
 * Box-Muller Transform, said noise is tuned using the given values of
 * mu (mean) and sigma2 (variance).
 * The same noise value is added/subtracted to each color channel of the image.
 */
void gaussian(PPM_Image *img, double mu, double sigma2)
{
    int i, r_sym, g_noise;
    double sample1, sample2;
    double boxmul_transform;
    double std_dev = sqrt(sigma2);
    const int img_size = img->header.width * img->header.height;
    for (i = 0; i <= img_size; i++) {
        // Randomly choses if the noise will be subtracted or added.
        r_sym = random() < 0.5 ? -1 : 1;

        sample1 = random();
        sample2 = random();
        boxmul_transform =
            ((sqrt(-2 * log2(sample1)) * cos(2 * PI * sample2) * std_dev) + mu);
        g_noise = (int)(boxmul_transform);

        img->pixelArray[i].red =
            MAX(0, MIN(255, img->pixelArray[i].red + (r_sym * g_noise)));
        img->pixelArray[i].green =
            MAX(0, MIN(255, img->pixelArray[i].green + (r_sym * g_noise)));
        img->pixelArray[i].blue =
            MAX(0, MIN(255, img->pixelArray[i].blue + (r_sym * g_noise)));
    }
}

int main(int argc, char *argv[])
{
    FILE *in;
    PPM_Image *img;
    double mu, sigma2;
    srand(time(0));

    // Input
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <imagepath> <mean> <variance>\n", argv[0]);
        return 1;
    }
    if ((in = fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "FATAL: Image file '%s' not found.\n", argv[1]);
        return 1;
    }
    if (sscanf(argv[2], "%lf", &mu) != 1) {
        fprintf(stderr, "FATAL: Error while reading the mean's value.\n");
        fclose(in);
        return 1;
    }
    if (sscanf(argv[3], "%lf", &sigma2) != 1) {
        fprintf(stderr, "FATAL: Error while reading the variance's value.\n");
        fclose(in);
        return 1;
    }
    if (sigma2 <= 0 || mu < 0) {
        fprintf(stderr, "FATAL: Incorrect values for mean or variance: mu >= 0 "
                        "& sigma2 > 0.\n");
        fclose(in);
        return 1;
    }

    img = malloc(sizeof(PPM_Image));
    assert(img != NULL);
    if (load_PPM(in, img) == 1) {
        fprintf(stderr, "FATAL: Error while loading image.\n");
        free(img);
        fclose(in);
        return 1;
    }
    fclose(in);

    gaussian(img, mu, sigma2);
    assert(img != NULL);
    if (write_PPM("noised_image.ppm", img) == 1) {
        fprintf(stderr, "FATAL: Error while writing noised image.\n");
        free(img);
        return 1;
    }

    free(img);
    return 0;
}