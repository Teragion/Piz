#ifndef LIGHT_H
#define LIGHT_H

#include "vector.h"

enum light_type {POINT, SPOT, DIRECTIONAL}; 

struct light {
    light_type type; 
    color col; 
    float intensity; // (0,1)
};

struct point_light : light {
    vec4 pos; 
};

struct directional_light :light {
    vec4 dir; 
}; 

#endif 
