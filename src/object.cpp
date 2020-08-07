#include <math.h>

#include "object.h"

void obj::add_vert(vec4 v) {
    num_vertices++;
    vlist_local.push_back(v);
    vlist_trans.push_back(v);
}

void obj::add_poly(polygon p) {
    num_polygons++;
    plist.push_back(p);
}

void obj::calc_max_avg_radius() {
    max_radius = 0;
    avg_radius = 0;

    for (auto it = vlist_local.begin(); it != vlist_local.end(); it++) {
        double dist = sqrt(it->x * it->x + it->y * it->y + it->z * it->z);
        if(dist > max_radius) 
            max_radius = dist;
        avg_radius += dist; 
    }
    
    avg_radius /= num_vertices;
}

mat44 obj::build_trans_mat() {
    mat44 ret = IDENTITY44;
    
    *ret(0, 3) = pos.x;
    *ret(1, 3) = pos.y;
    *ret(2, 3) = pos.z;

    return ret; 
}

void obj::convert_from_homogenous4d() {
    for (auto it = vlist_trans.begin(); it != vlist_trans.end(); it++) {
        vec4_divide_by_w(&*it);
    }
}

void obj::trans_vlist(mat44 *m, trans_mode mode) {
    vec4 tmp; 
    for (uint i = 0; i < num_vertices; i++) {
        switch(mode) {
            case LOCAL_ONLY:
            case LOCAL_TO_TRANS:
                mat44_mul(m, &vlist_local[i], &tmp);
                break;
            case TRANS_ONLY:
                mat44_mul(m, &vlist_trans[i], &tmp);
                break;
        }
        switch(mode) {
            case LOCAL_ONLY:
                vlist_local[i] = tmp;
                break;
            case LOCAL_TO_TRANS:
            case TRANS_ONLY:
                vlist_trans[i] = tmp;
                break;
        }
    }
}
