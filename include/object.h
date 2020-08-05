#ifndef OBJECT_H
#define OBJECT_H

#include <vector> 

#include "macros.h"
#include "polygon.h"
#include "vector.h"

struct obj {
    uint id; 
    char name [64];
    int state; 

    float avg_radius;  

    vec4 pos; 
    vec4 ux, uy, uz; 

    uint num_vertices; 

    std::vector<vec4> vlist_local; 
    std::vector<vec4> vlist_trans; 

    uint num_polygons; 

    std::vector<polygon> plist; 
};

#endif 
