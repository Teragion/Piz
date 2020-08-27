#include <math.h>

#include "camera.h"
#include "macros.h"

void camera_init(camera* cam,
                 vec4 pos, 
                 vec4 target,
                 float near_z,
                 float far_z, 
                 float fov,
                 float v_width,
                 float v_height) {
    // data copy 
    cam->pos = pos; 
    cam->target = target; 
    cam->near_clip_z = near_z; 
    cam->far_clip_z = far_z; 
    cam->fov = fov; 
    cam->viewport_width = v_width;
    cam->viewport_height = v_height; 

    // simple setup 
    cam->aspect = cam->viewport_width / cam->viewport_height;
    cam->viewplane_width = 2.0; 
    cam->viewplane_height = 2.0 / cam->aspect; 

    float tan_for_div2 = tan(DEG_TO_RAD(fov / 2)); 
    cam->view_dist = 0.5 * (cam->viewplane_width) * tan_for_div2; 

    if(fov == 90.0) { // simple case 
        vec3 p0; 
        vec3_init(&p0, 0, 0, 0); 

        vec3 n; 

        // right 
        vec3_init(&n, 1, 0, -1);    // x=z
        plane3_init(&cam->rt_clip_plane, p0, n); 

        // left 
        vec3_init(&n, -1, 0, -1);   // x=-z
        plane3_init(&cam->lt_clip_plane, p0, n); 

        // top 
        vec3_init(&n, 0, 1, -1);    // y=z
        plane3_init(&cam->tp_clip_plane, p0, n); 

        // bottom
        vec3_init(&n, 0, -1, -1);   // y=-z
        plane3_init(&cam->bt_clip_plane, p0, n); 
    } // TODO: handle not simple case
}

void camera_build_mcam(camera* cam) {
    mat44 mtrans_inv;
    mat44 muvn; // uvn projection matrix 
    mat44 mtmp; 

    mat44_init(&mtrans_inv, 1,  0,  0,  0,
                            0,  1,  0,  0,
                            0,  0,  1,  0,
                            -cam->pos.x, -cam->pos.y, -cam->pos.z, 1);

    // compute n vector 
    cam->n = cam->target; 
    vec4_sub(&cam->n, &cam->pos);

    vec4_init(&cam->v, 0, 1, 0); 

    // u = v x n 
    vec4_cross(&cam->v, &cam->n, &cam->u); 

    // v = n x u
    vec4_cross(&cam->n, &cam->u, &cam->v); 
    
    // normalize 
    vec4_normalize(&cam->u);
    vec4_normalize(&cam->v);
    vec4_normalize(&cam->n);

    mat44_init_col(&muvn, cam->u, cam->v, cam->n); 

    mat44_mul(&mtrans_inv, &muvn, &cam->mcam); 
}

void camera_build_mperspect(camera* cam) {
    // guru p.697
    mat44_init(&cam->mperspect, 
               cam->view_dist,   0,  0,  0,
               0,   cam->view_dist*cam->aspect,  0,  0,
               0,   0,  1,  1,
               0,   0,  0,  0);
}

void camera_build_mscr(camera* cam) {
    // guru p. 704
    // note: apply convert from homogenous 4d first 

    float alpha = 0.5 * cam->viewport_width - 0.5;
    float beta = 0.5 * cam->viewport_height - 0.5; 

    mat44_init(&cam->mscr, 
               alpha,   0,  0,  0,
               0,   -beta,  0,  0,
               alpha,   beta,   1,  0,
               0,   0,  0,  1);
}
