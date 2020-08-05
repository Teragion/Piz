#ifndef IMAGE_H
#define IMAGE_H

#include "macros.h"

enum format {
    FORMAT_LDR,
    FORMAT_HDR
};

struct image {
    format fmt;
    uint width, height, channels;
    unsigned char *ldr_buffer;
    float *hdr_buffer;
};

/* image creating/releasing */
image *image_create(uint width, uint height, uint channels, format fmt);
void image_release(image *ima);
image *image_load(const char *filename);
void image_save(image *img, const char *filename);

/* image processing */
void image_flip_h(image *img);
void image_flip_v(image *img);

#endif
