#include "plane.h"


void plane3_init(plane3* p, vec3 p0, vec3 n) {
    p->p0 = p0; 
    p->n = n;
}