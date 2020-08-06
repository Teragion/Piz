#ifndef CAMERA_H
#define CAMERA_H

#include "matrix.h"
#include "plane.h"
#include "vector.h"

// UVN camera 
struct camera {
    vec4 u; // right vector 
    vec4 v; // up vector 
    vec4 n; // target vector 

    vec4 target; 
    vec4 pos; 

    // clipping 
    float near_clip_z; 
    float far_clip_z; 

    float fov;

    plane3 rt_clip_plane; 
    plane3 lt_clip_plane; 
    plane3 tp_clip_plane; 
    plane3 bt_clip_plane;

    float viewport_width; 
    float viewport_height; 
    float aspect;

    float viewplane_width; 
    float viewplane_height;  
    float view_dist; 

    // projection matrices 
    mat44 mcam; 
    mat44 mperspect;
};

void camera_init(camera* cam,
                 vec4 pos, 
                 vec4 lookat,
                 float near_z,
                 float far_z, 
                 float fov,
                 float v_width,
                 float v_height); 

void camera_build_mcam(camera* cam); 
void camera_build_mperspect(camera* cam); 

#endif 
