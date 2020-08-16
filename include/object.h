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
enum illum_type {DIFFUSE, MIRROR, EMIT, TRANSPARENT}; // used in path tracer demo 

struct material; // avoid circular dependency

struct obj {
    uint id; 
    char name [64];
    int state; 
    obj_type type; 

    illum_type itype; 
    color albedo; // https://en.wikipedia.org/wiki/Albedo

    float max_radius; // culling 
    float avg_radius; // collision detection

    vec4 pos; 

    std::shared_ptr<material> obj_mat; 

    obj() : id(0), 
        name("unamed"),
        state(0),
        type(POLYHEDRON),
        itype(TRANSPARENT),
        albedo({0, 0, 0}),
        max_radius(0),
        avg_radius(0),
        pos({0, 0, 0, 1}),
        obj_mat(nullptr) {}
    
    mat44 build_trans_mat();

    virtual illum_type get_surface(vec4 *i_pnt, vec2 uv, uint trig_index, vec4 &normal, color &albedo_out) {
        return TRANSPARENT;
    } 
};

struct polyhed : obj{
    vec4 ux, uy, uz; // orientation 

    uint num_vertices; 

    std::vector<vec4> vlist_local; 
    std::vector<vec4> vlist_trans; 

    uint num_polygons; 

    std::vector<polygon*> plist; // maybe use unique pointers here? 

    polyhed() : num_vertices(0),
        num_polygons(0) {}

    void init(); 

    void add_vert(vec4 v);
    void add_poly(polygon *p);
    void calc_max_avg_radius();

    void convert_from_homogenous4d();
    void trans_vlist(mat44 *m, trans_mode mode);
};

struct trig_mesh : polyhed {
    std::vector<trig*> tlist; // maybe use unique pointers here? 

    trig_mesh() {
        type = TRIG_MESH;
    }

    void init_trig(); 

    void add_trig(int v0, int v1, int v2);

    /**
     * @param m 
     * @param v0 top left 
     * @param v1 top right 
     * @param v2 bot left 
     * @param v3 bot right 
     */
    void add_rect(int v0, int v1, int v2, int v3); 
    virtual illum_type get_surface(vec4 *i_pnt, vec2 uv, uint trig_index, vec4 &normal, color &albedo_out); 
};

struct sphere : obj{
    float radius; 

    sphere() {
        type = SPHERE; 
    }
    
    virtual illum_type get_surface(vec4 *i_pnt, vec2 uv, uint trig_index, vec4 &normal, color &albedo_out); 

};

struct ray {
    vec4 src; 
    vec4 dir; // normalized direction vector  
};


#endif 
