#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "image.h"
#include "macros.h"
#include "vector.h"

struct framebuffer {
    uint width, height;
    unsigned char *color_buffer;
    float *depth_buffer;
};

framebuffer* framebuffer_create(int width, int height); 
void framebuffer_release(framebuffer *fb); 

// clear color buffer
void framebuffer_ccolor(framebuffer *fb, vec4 color);

// clear depth buffer
void framebuffer_cdepth(framebuffer *fb, float depth); 

void blit_bgr(framebuffer *src, image *dst); 
void blit_rgb(framebuffer *src, image *dst); 

#endif
