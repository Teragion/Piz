#ifndef PLANE_H
#define PLANE_H

#include "vector.h"

struct plane3 {
    vec3 p0;    // any point 
    vec3 n;     // normal 
};

void plane3_init(plane3* p, vec3 p0, vec3 n);

#endif 
