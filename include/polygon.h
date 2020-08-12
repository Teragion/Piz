#ifndef POLYGON_H
#define POLYGON_H

#include <vector>

#include "vector.h"
#include "macros.h"

struct polygon {
    uint num_vertices; 
    std::vector<uint> vlist; 
};

struct trig : polygon {
    vec4 *v0_local; 
    vec4 *v1_local; 
    vec4 *v2_local;     

    vec4 *v0_trans; 
    vec4 *v1_trans; 
    vec4 *v2_trans;

    // normal vectors (normalized)
    vec4 n0; 
    vec4 n1; 
    vec4 n2;  
};

#endif 
