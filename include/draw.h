#ifndef DRAW_H
#define DRAW_H

#include "framebuffer.h"
#include "vector.h"

void draw_point(int x, int y, unsigned char color[4], framebuffer* fb);

void draw_line(int x1, int y1, int x2, int y2, unsigned char color[4], framebuffer* fb);

void draw_clip_line(int x1, int y1, int x2, int y2, unsigned char color[4], framebuffer* fb); 

int clip_line(int &x1,int &y1,int &x2, int &y2, framebuffer* fb);

#endif 
