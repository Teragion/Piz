#ifndef DRAW_H
#define DRAW_H

#include "framebuffer.h"
#include "object.h"
#include "polygon.h"
#include "vector.h"

// basic drawing operations 
void draw_point(int x, int y, color c, framebuffer* fb);
void draw_point(int x, int y, unsigned char color[4], framebuffer* fb);

void draw_line(int x1, int y1, int x2, int y2, unsigned char color[4], framebuffer* fb);

void draw_clip_line(int x1, int y1, int x2, int y2, unsigned char color[4], framebuffer* fb); 
void draw_clip_line(vec4 p1, vec4 p2, unsigned char color[4], framebuffer* fb); 

int clip_line(int &x1,int &y1,int &x2, int &y2, framebuffer* fb);

// apis 
void draw_polyhed_wireframe(polyhed *o, unsigned char color[4], framebuffer* fb); 

#endif 
