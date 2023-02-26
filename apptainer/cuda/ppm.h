/* Pietro Pezzi - pietro.pezzi3@studio.unibo.it - 0000925022 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    unsigned char red, green, blue;
} RGB_Pixel;

typedef struct {
    char type;
    int format;
    int width;
    int height;
    int max_value;
} PPM_Header;

typedef struct {
    PPM_Header header;
    RGB_Pixel *pixelArray;
} PPM_Image;

/*
 * Given a file pointer and a PPM_Image pointer, the function checks if
 * the given file is of supported PPM format by reading its header.
 * If the file is accepted, the content of the image is then loaded in the
 * given PPM_Image pointer.
 * Returns 0 with no errors, 1 otherwise.
 *
 * The current implementation does NOT check for comments in the header.
 */
int load_PPM(FILE *fp, PPM_Image *img)
{
    // Checking header.
    if (fscanf(fp, "%c%d\n", &img->header.type, &img->header.format) == EOF) {
        printf("Error while reading file type\n");
        return 1;
    }
    if (img->header.type != 'P' || img->header.format != 6) {
        printf("File is not of supported ppm format.\n");
        return 1;
    }
    if (fscanf(fp, "%d %d\n", &img->header.width, &img->header.height) == EOF) {
        printf("Error while reading image's size.\n");
        return 1;
    }
    if (fscanf(fp, "%d\n", &img->header.max_value) == EOF) {
        printf("Error while reading image's max value.\n");
        return 1;
    }
    if (img->header.max_value > 255) {
        printf("Max value grater than 255 is not supported.\n");
        return 1;
    }

    // Image
    img->pixelArray =
        (RGB_Pixel*)malloc(img->header.width * img->header.height * sizeof(RGB_Pixel));
    assert(img != NULL);
    if (fread(img->pixelArray, 3 * img->header.width, img->header.height, fp) !=
        img->header.height) {
        printf("Error while reading pixel array.\n");
        return 1;
    }

    return 0;
}

/*
 * Given a filename and a PPM_Image pointer, this function creates an image
 * file with the given PPM_Image contents.
 * Returns 0 with no errors, 1 otherwise.
 */
int write_PPM(const char *filename, PPM_Image *img)
{
    int i;
    const int img_size = img->header.width * img->header.height;
    FILE *fp;
    if ((fp = fopen(filename, "wb")) == NULL) {
        printf("Could not create file '%s'.\n", filename);
        fclose(fp);
        return 1;
    }
    // Header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", img->header.width, img->header.height);
    fprintf(fp, "255\n");

    // Image
    for (i = 0; i <= img_size; i++) {
        fprintf(fp, "%c%c%c", img->pixelArray[i].red, img->pixelArray[i].green,
                img->pixelArray[i].blue);
    }
    fclose(fp);
    return 0;
}