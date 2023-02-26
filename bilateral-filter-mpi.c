/* Pietro Pezzi - pietro.pezzi3@studio.unibo.it - 0000925022 */
#include "hpc.h"
#include "ppm.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
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
 * 
 * MPI implementation parameters:
 *    start_i: starting row index.
 *    start_j: starting column index.
 *    M, N: width and height of given portion.
 *    amount: amount of pixels to compute.
 */
void bilateral_filter(RGB_Pixel *pixelArray, RGB_Pixel *filtered, int start_i,
                      int start_j, int M, int N, int amount, int radius,
                      double sigma_color, double sigma_spatial)
{
    int i, j, k, l, count = 0;
    RGB_Double total_weight, partial_weight, neighborhood, gauss_color;
    double gauss_spatial;
    for (i = start_i; i < M; i++) {
        if (count >= amount) {
            break;
        }
        for (j = (i == start_i ? start_j : 0); j < N; j++) {
            if (count >= amount) {
                break;
            }
            set_dpixel(&total_weight, 0.0);
            set_dpixel(&partial_weight, 0.0);
            set_dpixel(&neighborhood, 0.0);
            for (k = -radius; k <= radius; k++) {
                for (l = -radius; l <= radius; l++) {
                    gauss_color.red =
                        gauss_kernel(pixelArray[IDX(i + k, j + l, M, N)].red -
                                         pixelArray[IDX(i, j, M, N)].red,
                                     sigma_color);
                    gauss_color.green =
                        gauss_kernel(pixelArray[IDX(i + k, j + l, M, N)].green -
                                         pixelArray[IDX(i, j, M, N)].green,
                                     sigma_color);
                    gauss_color.blue =
                        gauss_kernel(pixelArray[IDX(i + k, j + l, M, N)].blue -
                                         pixelArray[IDX(i, j, M, N)].blue,
                                     sigma_color);
                    gauss_spatial = gauss_kernel(eucl_dist(i + k, j + l, i, j),
                                                 sigma_spatial);
                    partial_weight.red = gauss_color.red * gauss_spatial;
                    partial_weight.green = gauss_color.green * gauss_spatial;
                    partial_weight.blue = gauss_color.blue * gauss_spatial;
                    neighborhood.red +=
                        (pixelArray[IDX(i + k, j + l, M, N)].red *
                         partial_weight.red);
                    neighborhood.green +=
                        (pixelArray[IDX(i + k, j + l, M, N)].green *
                         partial_weight.green);
                    neighborhood.blue +=
                        (pixelArray[IDX(i + k, j + l, M, N)].blue *
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
            count++;
        }
    }
}

int main(int argc, char *argv[])
{
    FILE *in = NULL;
    PPM_Image *img = NULL;
    RGB_Pixel *grid = NULL;
    int M, N, i;
    int radius, my_rank, comm_sz;
    double sigma_color, sigma_spatial;
    double t1 = 0, t2 = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Rank 0 process checks inputs and loads image
    if (my_rank == 0) {
        if (argc != 5) {
            fprintf(
                stderr,
                "USAGE: <imagepath> <radius> <sigma_color> <sigma_spatial>.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if ((in = fopen(argv[1], "rb")) == NULL) {
            fprintf(stderr, "FATAL: Image file '%s' not found.\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (sscanf(argv[2], "%d", &radius) != 1) {
            fprintf(stderr, "FATAL: Error while reading the radius's value.\n");
            fclose(in);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (sscanf(argv[3], "%lf", &sigma_color) != 1) {
            fprintf(stderr,
                    "FATAL: Error while reading the sigma-color's value.\n");
            fclose(in);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (sscanf(argv[4], "%lf", &sigma_spatial) != 1) {
            fprintf(stderr,
                    "FATAL: Error while reading the sigma-spatial's value.\n");
            fclose(in);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        img = malloc(sizeof(PPM_Image));
        assert(img != NULL);
        if (load_PPM(in, img) == 1) {
            fprintf(stderr, "FATAL: Error while loading image.\n");
            fclose(in);
            free(img);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fclose(in);
        if(radius > floor(img->header.height/comm_sz)-1){
            fprintf(stderr, "FATAL: incorrect value for radius.\n");
            free(img);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        const size_t array_size =
            (img->header.width * img->header.height) * sizeof(RGB_Pixel);
        grid = malloc(array_size * sizeof(*grid));
        assert(grid != NULL);

        grid = img->pixelArray;

        M = img->header.height;
        N = img->header.width;
    }

    if (my_rank == 0){
        t1 = hpc_gettime();
    }

    // The image's width and height is broadcasted to all processes.
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All processes read the needed bilateral filtering parameters.
    sscanf(argv[2], "%d", &radius);
    sscanf(argv[3], "%lf", &sigma_color);
    sscanf(argv[4], "%lf", &sigma_spatial);

    // Create MPI Datatype from RGB_Pixel struct.
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR,
                             MPI_UNSIGNED_CHAR};
    MPI_Datatype MPI_PIXEL;
    MPI_Aint offsets[3];

    offsets[0] = offsetof(RGB_Pixel, red);
    offsets[1] = offsetof(RGB_Pixel, green);
    offsets[2] = offsetof(RGB_Pixel, blue);

    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);

    // Computes: counts, displacements and previous/post ghost pixels.
    int counts[comm_sz];
    int displs[comm_sz];
    int prev_ghost[comm_sz];
    int post_ghost[comm_sz];
    int max_portion = (ceil((M * N) / (double)comm_sz));
    int rem_pixels = M * N, sum_counts = 0, amount, I_end;
    for (i = 0; i < comm_sz; i++) {
        amount = rem_pixels - max_portion >= 0 ? max_portion : rem_pixels;
        counts[i] = amount;
        rem_pixels =
            rem_pixels - max_portion >= 0 ? rem_pixels - max_portion : 0;
        displs[i] = sum_counts;
        sum_counts += counts[i];

        prev_ghost[i] = (radius * N) + (displs[i] - floor(displs[i] / N) * N);
        I_end = (displs[i] + counts[i]) - 1;
        post_ghost[i] = 
            (radius * N) + ((((floor(I_end / N) + 1) * N) - 1) - I_end);
    }

    // Compute next and previous rank
    int next_rank = my_rank + 1 >= comm_sz ? 0 : my_rank + 1;
    int prev_rank = my_rank - 1 < 0 ? comm_sz - 1 : my_rank - 1;

    const size_t portion_size =
        (prev_ghost[my_rank] + counts[my_rank] + post_ghost[my_rank]) 
        * sizeof(RGB_Pixel);
    RGB_Pixel *my_portion = malloc(portion_size * sizeof(*my_portion));
    assert(my_portion != NULL);

    // Scatters computable pixels.
    MPI_Scatterv(grid,                             // sendbuf 
                 counts,                           // sendcounts
                 displs,                           // displacements 
                 MPI_PIXEL,                        // sendtype
                 &my_portion[prev_ghost[my_rank]], // recvbuf
                 counts[my_rank],                  // recvcount
                 MPI_PIXEL,                        // recvtype
                 0,                                // root  
                 MPI_COMM_WORLD);                  // comm

    // Send and receive previous ghost pixels.
    MPI_Sendrecv(&my_portion[prev_ghost[my_rank] + counts[my_rank] 
                     - prev_ghost[next_rank]],     // sendbuf
                 prev_ghost[next_rank],            // sendcount
                 MPI_PIXEL,                        // sendtype
                 next_rank,                        // dest
                 0,                                // sendtag
                 my_portion,                       // recvbuf
                 prev_ghost[my_rank],              // recvcount
                 MPI_PIXEL,                        // recvtype
                 prev_rank,                        // source 
                 0,                                // recvtag
                 MPI_COMM_WORLD,                   // comm 
                 MPI_STATUS_IGNORE);               // status

    // Send and receive post ghost pixels.
    MPI_Sendrecv(&my_portion[prev_ghost[my_rank]], 
                 post_ghost[prev_rank], 
                 MPI_PIXEL,
                 prev_rank, 
                 0, 
                 &my_portion[prev_ghost[my_rank] + counts[my_rank]], 
                 post_ghost[my_rank],
                 MPI_PIXEL, 
                 next_rank, 
                 0, 
                 MPI_COMM_WORLD, 
                 MPI_STATUS_IGNORE);

    // Apply bilateral filtering on portion.
    int start_i = floor(prev_ghost[my_rank] / N);
    int start_j = prev_ghost[my_rank] - start_i * N;

    const size_t filtred_size =
        (prev_ghost[my_rank] + counts[my_rank]) * sizeof(RGB_Pixel);
    RGB_Pixel *filtered = malloc(filtred_size * sizeof(*filtered));
    assert(filtered != NULL);

    bilateral_filter(my_portion, filtered, start_i, start_j,
                     (prev_ghost[my_rank] + counts[my_rank] 
                     + post_ghost[my_rank] / N), N,
                     counts[my_rank], radius, sigma_color, sigma_spatial);

    // Gather results.
    MPI_Gatherv(&filtered[prev_ghost[my_rank]],    // sendbuf
                counts[my_rank],                   // sendcount
                MPI_PIXEL,                         // sendtype
                grid,                              // recvbuf
                counts,                            // recvcounts
                displs,                            // displacements
                MPI_PIXEL,                         // recvtype
                0,                                 // root 
                MPI_COMM_WORLD);                   // comm

    if (my_rank == 0){
        t2 = hpc_gettime();
    }

    // Write final image
    if (my_rank == 0) {
        img->pixelArray = grid;
        if (write_PPM("filtered_image_mpi.ppm", img) == 1) {
            fprintf(stderr, "FATAL: Error while writing filtered image.\n");
            free(filtered);
            free(my_portion);
            free(grid);
            free(img);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        free(grid);
        free(img);

        printf("MPI parallel region execution time: %fs\n", t2-t1);
    }

    free(filtered);
    free(my_portion);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
