#ifndef OBJECT_H
#define OBJECT_H

#include <vector> 

#include "macros.h"
#include "matrix.h"
#include "polygon.h"
#include "vector.h"

enum trans_mode {LOCAL_TO_TRANS, TRANS_ONLY, LOCAL_ONLY};

struct obj {
    uint id; 
    char name [64];
    int state; 

    float max_radius; // culling 
    float avg_radius; // collision detection

    vec4 pos; 
    vec4 ux, uy, uz; // orientation 

    uint num_vertices; 

    std::vector<vec4> vlist_local; 
    std::vector<vec4> vlist_trans; 

    uint num_polygons; 

    std::vector<polygon> plist; 

    void add_vert(vec4 v);
    void add_poly(polygon p);
    void calc_max_avg_radius();

    mat44 build_trans_mat();

    void convert_from_homogenous4d();
    void trans_vlist(mat44 *m, trans_mode mode);
};

#endif 
