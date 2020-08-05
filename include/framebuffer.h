#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "macros.h"

struct framebuffer {
    uint width, height;
    unsigned char *color_buffer;
    float *depth_buffer;
};

#endif
