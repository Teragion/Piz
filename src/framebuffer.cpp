#include <algorithm>
#include <assert.h>
#include <stdlib.h>

#include "framebuffer.h"
#include "maths.h"

framebuffer* framebuffer_create(int width, int height) {
    assert(width > 0 && height > 0); 

    uint cbuffer_size = width * height * 4; 
    uint dbuffer_size = sizeof(float) * width * height;

    vec4 default_color = {0, 0, 0, 1}; 
    float default_depth = 1.0; 

    framebuffer* fb = (framebuffer*)malloc(sizeof(framebuffer)); 
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
    for(uint i = 0; i < bsize; i++) {
        std::fill_n(fb->depth_buffer, bsize, depth);
    } 
}
