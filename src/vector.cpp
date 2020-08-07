#include <math.h>

#include "vector.h"
#include "macros.h"

void vec3_normalize(vec3* v) {
    float length = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);

    if(length < EPSI_E5) // does no operation for zero vector 
        return; 

    float length_inv = 1.0 / length; 
    v->x *= length_inv; 
    v->y *= length_inv; 
    v->z *= length_inv; 
}

void vec4_normalize(vec4* v) {
    float length = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);

    if(length < EPSI_E5) // does no operation for zero vector 
        return; 

    float length_inv = 1.0 / length; 
    v->x *= length_inv; 
    v->y *= length_inv; 
    v->z *= length_inv; 
    v->w = 1;
}

void vec4_divide_by_w(vec4* v) {
    double w_inv = 1/v->w; 
    v->x *= w_inv;
    v->y *= w_inv;    
    v->z *= w_inv;
    v->w = 1.0;    
}

void vec4_cross(vec4* a, vec4* b, vec4* res) {
    res->x =  ( (a->y * b->z) - (a->z * b->y) );
    res->y = -( (a->x * b->z) - (a->z * b->x) );
    res->z =  ( (a->x * b->y) - (a->y * b->x) ); 
    res->w = 1;
}
