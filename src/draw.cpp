#include <algorithm>
#include <math.h>

#include "draw.h"
#include "maths.h"

static void swap_int(int *a, int *b) {
    int t = *a;
    *a = *b; 
    *b = t; 
}

static inline int lerp_int(int a, int b, float t) {
    return (a + (b - a) * t);
}

unsigned char proc[4];

void draw_point(int x, int y, color c, framebuffer* fb) {
    proc[0] = std::clamp(c.x, 0.f, 1.f) * 255;
    proc[1] = std::clamp(c.y, 0.f, 1.f) * 255;    
    proc[2] = std::clamp(c.z, 0.f, 1.f) * 255;
    proc[3] = 255; // opaque 

    draw_point(x, y, proc, fb);
}

void draw_point(int x, int y, unsigned char color[4], framebuffer* fb) {
    int index = (y * fb->width + x) * 4;
    for (int i = 0; i < 4; i++) {
        fb->color_buffer[index + i] = color[i];
    }
}

void draw_line(int x1, int y1, int x2, int y2, unsigned char color[4], framebuffer* fb) {
    // Bresenham's line algorithm 

    int x_dist = abs(x2 - x1);
    int y_dist = abs(y2 - y1);
    if (x_dist == 0 && y_dist == 0) {
        draw_point(x1, y1, color, fb);
    } else if (x_dist > y_dist) {
        if (x1 > x2) {
            swap_int(&x1, &x2);
            swap_int(&y1, &y2);
        }
        for (int x = x1; x <= x2; x++) {
            float t = (float)(x - x1) / (float)x_dist;
            int y = lerp_int(y1, y2, t);
            draw_point(x, y, color, fb);
        }
    } else {
        if (y1 > y2) {
            swap_int(&x1, &x2);
            swap_int(&y1, &y2);
        }
        for (int y = y1; y <= y2; y++) {
            float t = (float)(y - y1) / (float)y_dist;
            int x = lerp_int(x1, x2, t);
            draw_point(x, y, color, fb);
        }
    }
}

void draw_clip_line(int x1, int y1, int x2, int y2, unsigned char color[4], framebuffer* fb) {
    int cxs, cys,
        cxe, cye;

    // clip and draw each line
    cxs = x1;
    cys = y1;
    cxe = x2;
    cye = y2;

    // clip the line
    if (clip_line(cxs, cys, cxe, cye, fb))
        draw_line(cxs, cys, cxe, cye, color, fb);
}

void draw_clip_line(vec4 p1, vec4 p2, unsigned char color[4], framebuffer *fb) {
    draw_clip_line(p1.x, p1.y, p2.x, p2.y, color, fb);
}

int clip_line(int &x1, int &y1, int &x2, int &y2, framebuffer* fb) {
    // this function clips the sent line using the globally defined clipping
    // region

    int min_clip_x = 0,
        min_clip_y = 0,
        max_clip_x = fb->width - 1,
        max_clip_y = fb->height - 1;

    // internal clipping codes
#define CLIP_CODE_C  0x0000
#define CLIP_CODE_N  0x0008
#define CLIP_CODE_S  0x0004
#define CLIP_CODE_E  0x0002
#define CLIP_CODE_W  0x0001

#define CLIP_CODE_NE 0x000a
#define CLIP_CODE_SE 0x0006
#define CLIP_CODE_NW 0x0009 
#define CLIP_CODE_SW 0x0005

    int xc1=x1, 
        yc1=y1, 
        xc2=x2, 
        yc2=y2;

    int p1_code=0, 
        p2_code=0;

    // determine codes for p1 and p2
    if (y1 < min_clip_y)
        p1_code|=CLIP_CODE_N;
    else
        if (y1 > max_clip_y)
            p1_code|=CLIP_CODE_S;

    if (x1 < min_clip_x)
        p1_code|=CLIP_CODE_W;
    else
        if (x1 > max_clip_x)
            p1_code|=CLIP_CODE_E;

    if (y2 < min_clip_y)
        p2_code|=CLIP_CODE_N;
    else
        if (y2 > max_clip_y)
            p2_code|=CLIP_CODE_S;

    if (x2 < min_clip_x)
        p2_code|=CLIP_CODE_W;
    else
        if (x2 > max_clip_x)
            p2_code|=CLIP_CODE_E;

    // try and trivially reject
    if ((p1_code & p2_code)) 
        return(0);

    // test for totally visible, if so leave points untouched
    if (p1_code==0 && p2_code==0)
        return(1);

    // determine end clip point for p1
    switch(p1_code)
    {
    case CLIP_CODE_C: break;

    case CLIP_CODE_N:
        {
            yc1 = min_clip_y;
            xc1 = x1 + 0.5+(min_clip_y-y1)*(x2-x1)/(y2-y1);
        } break;
    case CLIP_CODE_S:
        {
            yc1 = max_clip_y;
            xc1 = x1 + 0.5+(max_clip_y-y1)*(x2-x1)/(y2-y1);
        } break;

    case CLIP_CODE_W:
        {
            xc1 = min_clip_x;
            yc1 = y1 + 0.5+(min_clip_x-x1)*(y2-y1)/(x2-x1);
        } break;

    case CLIP_CODE_E:
        {
            xc1 = max_clip_x;
            yc1 = y1 + 0.5+(max_clip_x-x1)*(y2-y1)/(x2-x1);
        } break;

        // these cases are more complex, must compute 2 intersections
    case CLIP_CODE_NE:
        {
            // north hline intersection
            yc1 = min_clip_y;
            xc1 = x1 + 0.5+(min_clip_y-y1)*(x2-x1)/(y2-y1);

            // test if intersection is valid, of so then done, else compute next
            if (xc1 < min_clip_x || xc1 > max_clip_x)
            {
                // east vline intersection
                xc1 = max_clip_x;
                yc1 = y1 + 0.5+(max_clip_x-x1)*(y2-y1)/(x2-x1);
            } // end if

        } break;

    case CLIP_CODE_SE:
        {
            // south hline intersection
            yc1 = max_clip_y;
            xc1 = x1 + 0.5+(max_clip_y-y1)*(x2-x1)/(y2-y1);	

            // test if intersection is valid, of so then done, else compute next
            if (xc1 < min_clip_x || xc1 > max_clip_x)
            {
                // east vline intersection
                xc1 = max_clip_x;
                yc1 = y1 + 0.5+(max_clip_x-x1)*(y2-y1)/(x2-x1);
            } // end if

        } break;

    case CLIP_CODE_NW: 
        {
            // north hline intersection
            yc1 = min_clip_y;
            xc1 = x1 + 0.5+(min_clip_y-y1)*(x2-x1)/(y2-y1);

            // test if intersection is valid, of so then done, else compute next
            if (xc1 < min_clip_x || xc1 > max_clip_x)
            {
                xc1 = min_clip_x;
                yc1 = y1 + 0.5+(min_clip_x-x1)*(y2-y1)/(x2-x1);	
            } // end if

        } break;

    case CLIP_CODE_SW:
        {
            // south hline intersection
            yc1 = max_clip_y;
            xc1 = x1 + 0.5+(max_clip_y-y1)*(x2-x1)/(y2-y1);	

            // test if intersection is valid, of so then done, else compute next
            if (xc1 < min_clip_x || xc1 > max_clip_x)
            {
                xc1 = min_clip_x;
                yc1 = y1 + 0.5+(min_clip_x-x1)*(y2-y1)/(x2-x1);	
            } // end if

        } break;

    default:break;

    } // end switch

    // determine clip point for p2
    switch(p2_code)
    {
    case CLIP_CODE_C: break;

    case CLIP_CODE_N:
        {
            yc2 = min_clip_y;
            xc2 = x2 + (min_clip_y-y2)*(x1-x2)/(y1-y2);
        } break;

    case CLIP_CODE_S:
        {
            yc2 = max_clip_y;
            xc2 = x2 + (max_clip_y-y2)*(x1-x2)/(y1-y2);
        } break;

    case CLIP_CODE_W:
        {
            xc2 = min_clip_x;
            yc2 = y2 + (min_clip_x-x2)*(y1-y2)/(x1-x2);
        } break;

    case CLIP_CODE_E:
        {
            xc2 = max_clip_x;
            yc2 = y2 + (max_clip_x-x2)*(y1-y2)/(x1-x2);
        } break;

        // these cases are more complex, must compute 2 intersections
    case CLIP_CODE_NE:
        {
            // north hline intersection
            yc2 = min_clip_y;
            xc2 = x2 + 0.5+(min_clip_y-y2)*(x1-x2)/(y1-y2);

            // test if intersection is valid, of so then done, else compute next
            if (xc2 < min_clip_x || xc2 > max_clip_x)
            {
                // east vline intersection
                xc2 = max_clip_x;
                yc2 = y2 + 0.5+(max_clip_x-x2)*(y1-y2)/(x1-x2);
            } // end if

        } break;

    case CLIP_CODE_SE:
        {
            // south hline intersection
            yc2 = max_clip_y;
            xc2 = x2 + 0.5+(max_clip_y-y2)*(x1-x2)/(y1-y2);	

            // test if intersection is valid, of so then done, else compute next
            if (xc2 < min_clip_x || xc2 > max_clip_x)
            {
                // east vline intersection
                xc2 = max_clip_x;
                yc2 = y2 + 0.5+(max_clip_x-x2)*(y1-y2)/(x1-x2);
            } // end if

        } break;

    case CLIP_CODE_NW: 
        {
            // north hline intersection
            yc2 = min_clip_y;
            xc2 = x2 + 0.5+(min_clip_y-y2)*(x1-x2)/(y1-y2);

            // test if intersection is valid, of so then done, else compute next
            if (xc2 < min_clip_x || xc2 > max_clip_x)
            {
                xc2 = min_clip_x;
                yc2 = y2 + 0.5+(min_clip_x-x2)*(y1-y2)/(x1-x2);	
            } // end if

        } break;

    case CLIP_CODE_SW:
        {
            // south hline intersection
            yc2 = max_clip_y;
            xc2 = x2 + 0.5+(max_clip_y-y2)*(x1-x2)/(y1-y2);	

            // test if intersection is valid, of so then done, else compute next
            if (xc2 < min_clip_x || xc2 > max_clip_x)
            {
                xc2 = min_clip_x;
                yc2 = y2 + 0.5+(min_clip_x-x2)*(y1-y2)/(x1-x2);	
            } // end if

        } break;

    default:break;

    } // end switch

    // do bounds check
    if ((xc1 < min_clip_x) || (xc1 > max_clip_x) ||
        (yc1 < min_clip_y) || (yc1 > max_clip_y) ||
        (xc2 < min_clip_x) || (xc2 > max_clip_x) ||
        (yc2 < min_clip_y) || (yc2 > max_clip_y) )
    {
        return(0);
    } // end if

    // store vars back
    x1 = xc1;
    y1 = yc1;
    x2 = xc2;
    y2 = yc2;

    return(1);
} 

void draw_polyhed_wireframe(polyhed *o, unsigned char color[4], framebuffer* fb) {
    for (auto it = o->plist.begin(); it != o->plist.end(); it++) {
        for (int i = 0; i < (*it)->num_vertices; i++) {
            draw_clip_line(o->vlist_trans[(*it)->vlist[i]], o->vlist_trans[(*it)->vlist[(i + 1) % (*it)->num_vertices]], color, fb);
        }
    }
}
