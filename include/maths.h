#ifndef MATHS_H
#define MATHS_H

#include <limits>
#include <vector>

#include "macros.h"
#include "object.h"

// Type definitions 

// vector/point

// matrix

// quaternion

// curve

// Functions 

// returns true if real solutions exist 
bool quadratic_solve(double a, double b, double c, float &x0, float &x1) {
    float discr = b * b - 4 * a * c; 
    if (discr < 0) return false; 
    else if (discr == 0) { 
        x0 = x1 = - 0.5 * b / a; 
    } 
    else { 
        float q = (b > 0) ? 
            -0.5 * (b + sqrt(discr)) : 
            -0.5 * (b - sqrt(discr)); 
        x0 = q / a; 
        x1 = c / q; 
    } 
 
    return true; 
}

// TODO: use sin/cos lookup tables to speedup computation. Actually, 
//       check if it actually increases performance 

// converts float in range (0, 1) to unsigned char 
inline unsigned char float_to_uchar(float x) {
    return (unsigned char)x * 255; 
}

// ray tracing related 

// stores intersection information 
struct isect {
    vec4 isect_pnt; 
    float inear; 

    obj* o;

    // for trig_mesh
    vec2 uv; 
    uint trig_index; 
};

void isect_init(isect &i) {
    i.inear = std::numeric_limits<float>::max(); 
}

// returns true if intersection detected
bool ray_sphere_intersect(ray *r, sphere *s, float &inear);

// returns true if intersection detected
bool ray_trig_intersect(ray *r, const trig *trig, float &inear, float &u, float &v);

// returns true if intersection detected
bool ray_trig_mesh_intersect(ray *r, trig_mesh *o, float &res, vec2 &uv, uint trig_index);

// trace light for all objects (only type = SPHERE||TRIG_MESH)
bool trace(ray *r, std::vector<obj*> obj_list, isect &res);

#endif
