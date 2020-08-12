#ifndef OBJECT_H
#define OBJECT_H

#include <memory>
#include <vector> 

#include "macros.h"
#include "matrix.h"
#include "polygon.h"
#include "vector.h"

enum trans_mode {LOCAL_TO_TRANS, TRANS_ONLY, LOCAL_ONLY};
enum obj_type {SPHERE, TRIG_MESH, POLYHEDRON}; 

struct obj {
    uint id; 
    char name [64];
    int state; 
    obj_type type; 

    float max_radius; // culling 
    float avg_radius; // collision detection

    vec4 pos; 
    
    mat44 build_trans_mat();
};

struct polyhed : obj{
    vec4 ux, uy, uz; // orientation 

    uint num_vertices; 

    std::vector<vec4> vlist_local; 
    std::vector<vec4> vlist_trans; 

    uint num_polygons; 

    std::vector<polygon*> plist; // maybe use unique pointers here? 

    void add_vert(vec4 v);
    void add_poly(polygon *p);
    void calc_max_avg_radius();

    void convert_from_homogenous4d();
    void trans_vlist(mat44 *m, trans_mode mode);
};

struct trig_mesh : polyhed {
    std::vector<trig*> tlist; // maybe use unique pointers here? 

    void add_trig(int v0, int v1, int v2);
};

struct sphere : obj{
    float radius; 
};

struct ray {
    vec4 src; 
    vec4 dir; // normalized direction vector  
};


#endif 
