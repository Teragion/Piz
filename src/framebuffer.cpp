#include <algorithm>
#include <assert.h>
#include <stdlib.h>

#include "framebuffer.h"
#include "macros.h"
#include "maths.h"

framebuffer* framebuffer_create(int width, int height) {
    assert(width > 0 && height > 0); 

    uint cbuffer_size = width * height * 4; 
    uint dbuffer_size = sizeof(float) * width * height;

    vec4 default_color = {0, 0, 0, 1}; 
    float default_depth = 1.0; 

    framebuffer* fb = (framebuffer*)malloc(sizeof(framebuffer)); 
    if (fb == NULL)
        return NULL;
    fb->width = width; 
    fb->height = height; 
    fb->color_buffer = (unsigned char*)malloc(cbuffer_size); 
    fb->depth_buffer = (float*)malloc(dbuffer_size);  

    framebuffer_ccolor(fb, default_color);
    framebuffer_cdepth(fb, default_depth);

    return fb; 
}

void framebuffer_release(framebuffer *fb) {
    free(fb->color_buffer);
    free(fb->depth_buffer);
    free(fb);
}

void framebuffer_ccolor(framebuffer *fb, vec4 color) {
    uint bsize = fb->width * fb->height; 
    unsigned char x = float_to_uchar(color.x);
    unsigned char y = float_to_uchar(color.y);
    unsigned char z = float_to_uchar(color.z);
    unsigned char w = float_to_uchar(color.w);

    for (uint i = 0; i < bsize; i++) {
        fb->color_buffer[i * 4 + 0] = x;
        fb->color_buffer[i * 4 + 1] = y;
        fb->color_buffer[i * 4 + 2] = z;
        fb->color_buffer[i * 4 + 3] = w;
    }
}

void framebuffer_cdepth(framebuffer *fb, float depth) {
    uint bsize = fb->width * fb->height;
    std::fill(fb->depth_buffer, fb->depth_buffer + bsize, depth);
}

void blit_bgr(framebuffer *src, image *dst) {
    int width = dst->width;
    int height = dst->height;
    int r, c;

    assert(src->width == dst->width && src->height == dst->height);
    assert(dst->fmt == FORMAT_LDR && dst->channels == 4);

    for (r = 0; r < height; r++) {
        for (c = 0; c < width; c++) {
            // int flipped_r = height - 1 - r;
            int src_index = (r * width + c) * 4;
            int dst_index = (r * width + c) * 4;
            unsigned char *src_pixel = &src->color_buffer[src_index];
            unsigned char *dst_pixel = &dst->ldr_buffer[dst_index];
            dst_pixel[0] = src_pixel[2];  /* blue */
            dst_pixel[1] = src_pixel[1];  /* green */
            dst_pixel[2] = src_pixel[0];  /* red */
        }
    }
}

void blit_rgb(framebuffer *src, image *dst) {
    int width = dst->width;
    int height = dst->height;
    int r, c;

    assert(src->width == dst->width && src->height == dst->height);
    assert(dst->fmt == FORMAT_LDR && dst->channels == 4);

    for (r = 0; r < height; r++) {
        for (c = 0; c < width; c++) {
            // int flipped_r = height - 1 - r;
            int src_index = (r * width + c) * 4;
            int dst_index = (r * width + c) * 4;
            unsigned char *src_pixel = &src->color_buffer[src_index];
            unsigned char *dst_pixel = &dst->ldr_buffer[dst_index];
            dst_pixel[0] = src_pixel[0];  /* red */
            dst_pixel[1] = src_pixel[1];  /* green */
            dst_pixel[2] = src_pixel[2];  /* blue */
        }
    }
}
