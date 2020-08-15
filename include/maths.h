#ifndef MATHS_H
#define MATHS_H

#include <limits>
#include <random>
#include <omp.h>
#include <vector>

#include "macros.h"
#include "object.h"

// Type definitions 

// vector/point

// matrix

// quaternion

// curve

// Functions 

// clampping 
void clamp(float& x);
void clamp(vec3& x);

// returns true if real solutions exist 
bool quadratic_solve(double a, double b, double c, float &x0, float &x1);

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

void isect_init(isect &i);

// returns true if intersection detected
bool ray_sphere_intersect(ray *r, sphere *s, float &inear);

// returns true if intersection detected
bool ray_trig_intersect(ray *r, const trig *trig, float &inear, float &u, float &v);

// returns true if intersection detected
bool ray_trig_mesh_intersect(ray *r, trig_mesh *o, float &res, vec2 &uv, uint &trig_index);

// trace light for all objects (only type = SPHERE||TRIG_MESH)
bool trace(ray *r, const std::vector<obj*> &obj_list, isect &res);

// random related 
extern std::default_random_engine _generator;
extern std::uniform_real_distribution<float> _dist01;

void random_init(uint seed = RANDOM_SEED); 

float random01();

float random02Pi();

/** generate spherically uniform vector and return direction vector in xyz coord 
 * TODO: study the sampling algorithm!
 * https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
 & @param r1 random number (0, 1)
 & @param r2 random number (0, 1)
 */
vec4 uniform_sample_hemis(const float &r1, const float &r2);

/**
 * @brief generate spherically uniform vector and return direction in xyz coord 
 * 
 * @param r1 random number (0, 2*Pi)
 * @param r2 random number (0, 1)
 * @return vec4 
 */
vec4 uniform_sample_sphere(const float &r1, const float &r2);

/** 
 * generates a matrix that transforms the vector generated in uniform_sample_hemis
 * from sample coord to world coord. 
 * @param N normal vector 
 */
mat44 create_sample_coord(const vec4 &N);



#endif
