#include <math.h>

#include "object.h"

void polyhed::init() {
    num_vertices = 0; 
    num_polygons = 0; 

    vlist_local = std::vector<vec4>(); 
    vlist_trans = std::vector<vec4>(); 
    plist = std::vector<polygon*>(); 
}

void polyhed::add_vert(vec4 v) {
    num_vertices++;
    vlist_local.push_back(v);
    vlist_trans.push_back(v);
}

void trig_mesh::init_trig() { 
    tlist = std::vector<trig*>(); 
}

void trig_mesh::add_trig(int v0, int v1, int v2) {
    // std::unique_ptr<trig> t = std::make_unique<trig>();
    trig *t = (trig*)malloc(sizeof(trig)); 
    t->num_vertices = 3;
    t->v0_local = &vlist_local[v0];
    t->v1_local = &vlist_local[v1];
    t->v2_local = &vlist_local[v2];    
    t->v0_trans = &vlist_trans[v0];
    t->v1_trans = &vlist_trans[v1];
    t->v2_trans = &vlist_trans[v2];
    num_polygons++; 
    tlist.push_back(t); 
}

void trig_mesh::add_rect(int v0, int v1, int v2, int v3) {
    add_trig(v0, v1, v2);
    add_trig(v1, v3, v2);
}

void polyhed::add_poly(polygon *p) {
    num_polygons++;
    // plist.push_back(std::make_unique<polygon>(*p));
    plist.push_back(p);
}

void polyhed::calc_max_avg_radius() {
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
    
    *ret(3, 0) = pos.x;
    *ret(3, 1) = pos.y;
    *ret(3, 2) = pos.z;

    return ret; 
}

void polyhed::convert_from_homogenous4d() {
    for (auto it = vlist_trans.begin(); it != vlist_trans.end(); it++) {
        vec4_divide_by_w(&*it);
    }
}

void polyhed::trans_vlist(mat44 *m, trans_mode mode) {
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

illum_type trig_mesh::get_surface(vec4 *i_pnt, vec2 uv, uint trig_index, vec4 &normal, color &albedo_out) {
    trig *t = tlist[trig_index]; 
    vec4 v0v1 = *t->v1_trans; 
    vec4_sub(&v0v1, t->v0_trans);
    vec4 v0v2 = *t->v2_trans; 
    vec4_sub(&v0v2, t->v0_trans);

    vec4_cross(&v0v1, &v0v2, &normal);
    vec4_normalize(&normal);

    albedo_out = albedo;

    return itype;
}

illum_type sphere::get_surface(vec4 *i_pnt, vec2 uv, uint trig_index, vec4 &normal, color &albedo_out) {
    normal = *i_pnt; 
    vec4_sub(&normal, &pos);
    vec4_normalize(&normal);
     
    albedo_out = albedo;

    return itype;
}
