#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "image.h"

/* image creating/releasing */
image *image_create(uint width, uint height, uint channels, format fmt) {
    uint num_elems = width * height * channels; 
    image *img; 

    assert(width > 0 && height > 0 && channels >= 1 && channels <= 4); 
    assert(fmt == FORMAT_LDR || fmt == FORMAT_HDR); 

    img = (image*) malloc(sizeof(image));
    img->fmt = fmt; 
    img->width = width; 
    img->height = height; 
    img->channels = channels;
    img->ldr_buffer = NULL;
    img->hdr_buffer = NULL; 

    uint size; 
    switch (fmt) {
    case FORMAT_LDR:
        size = sizeof(unsigned char) * num_elems; 
        img->ldr_buffer = (unsigned char*) malloc(size);
        break;
    
    case FORMAT_HDR:
        size = sizeof(float) * num_elems;
        img->hdr_buffer = (float*) malloc(size);
        break;

    default:
        break;
    }
    return img;
}

void image_release(image *img) {
    free(img->ldr_buffer); 
    free(img->hdr_buffer); 
    free(img); 
}

image *image_load(const char *filename);

static void save_tga_image(image *image, const char *filename);

void image_save(image *img, const char *filename) {
    if (img->fmt == FORMAT_LDR)
        save_tga_image(img, filename);
}

/* image processing */

static void swap_bytes(unsigned char *a, unsigned char *b) {
    unsigned char t = *a;
    *a = *b;
    *b = t;
}

static void swap_floats(float *a, float *b) {
    float t = *a;
    *a = *b;
    *b = t;
}

static unsigned char *get_ldr_pixel(image *img, int row, int col) {
    int index = (row * img->width + col) * img->channels;
    return &img->ldr_buffer[index];
}

static float *get_hdr_pixel(image *img, int row, int col) {
    int index = (row * img->width + col) * img->channels;
    return &img->hdr_buffer[index];
}

void image_flip_h(image *img) {
    uint half_width = img->width / 2;
    uint r, c, k;
    for (r = 0; r < img->height; r++) {
        for (c = 0; c < half_width; c++) {
            int flipped_c = img->width - 1 - c;
            if (img->fmt == FORMAT_LDR) {
                unsigned char *pixel1 = get_ldr_pixel(img, r, c);
                unsigned char *pixel2 = get_ldr_pixel(img, r, flipped_c);
                for (k = 0; k < img->channels; k++) {
                    swap_bytes(&pixel1[k], &pixel2[k]);
                }
            } else {
                float *pixel1 = get_hdr_pixel(img, r, c);
                float *pixel2 = get_hdr_pixel(img, r, flipped_c);
                for (k = 0; k < img->channels; k++) {
                    swap_floats(&pixel1[k], &pixel2[k]);
                }
            }
        }
    }
}

void image_flip_v(image *img) {
    int half_height = img->height / 2;
    int r, c, k;
    for (r = 0; r < half_height; r++) {
        for (c = 0; c < img->width; c++) {
            int flipped_r = img->height - 1 - r;
            if (img->fmt == FORMAT_LDR) {
                unsigned char *pixel1 = get_ldr_pixel(img, r, c);
                unsigned char *pixel2 = get_ldr_pixel(img, flipped_r, c);
                for (k = 0; k < img->channels; k++) {
                    swap_bytes(&pixel1[k], &pixel2[k]);
                }
            } else {
                float *pixel1 = get_hdr_pixel(img, r, c);
                float *pixel2 = get_hdr_pixel(img, flipped_r, c);
                for (k = 0; k < img->channels; k++) {
                    swap_floats(&pixel1[k], &pixel2[k]);
                }
            }
        }
    }
}

// TGA related
// https://github.com/zauonlok/renderer/blob/master/renderer/core/image.c

#define TGA_HEADER_SIZE 18

static void write_bytes(FILE *file, void *buffer, int size) {
    int count = (int)fwrite(buffer, 1, size, file);
    assert(count == size);
    (void)(count);
}

static int get_num_elems(image *image) {
    return image->width * image->height * image->channels;
}

static void save_tga_image(image *image, const char *filename) {
    unsigned char header[TGA_HEADER_SIZE];
    FILE *file;

    assert(image->fmt == FORMAT_LDR);

    file = fopen(filename, "wb");
    assert(file != NULL);

    memset(header, 0, TGA_HEADER_SIZE);
    header[2] = image->channels == 1 ? 3 : 2;               /* image type */
    header[12] = image->width & 0xFF;                       /* width, lsb */
    header[13] = (image->width >> 8) & 0xFF;                /* width, msb */
    header[14] = image->height & 0xFF;                      /* height, lsb */
    header[15] = (image->height >> 8) & 0xFF;               /* height, msb */
    header[16] = (image->channels * 8) & 0xFF;              /* image depth */
    write_bytes(file, header, TGA_HEADER_SIZE);

    if (image->channels >= 3) {
        int r, c;
        for (r = 0; r < image->height; r++) {
            for (c = 0; c < image->width; c++) {
                unsigned char *pixel = get_ldr_pixel(image, r, c);
                unsigned char channels[4];
                memcpy(channels, pixel, image->channels);
                swap_bytes(&channels[0], &channels[2]);     /* rgb to bgr */
                write_bytes(file, channels, image->channels);
            }
        }
    } else {
        write_bytes(file, image->ldr_buffer, get_num_elems(image));
    }

    fclose(file);
}