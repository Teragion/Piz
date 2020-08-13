#include "maths.h"

bool quadratic_solve(double a, double b, double c, float &x0, float &x1) {
    float discr = b * b - 4 * a * c; 
    if (discr < 0) return false; 
    else if (discr == 0) { 
        x0 = x1 = - 0.5 * b / a; 
    } 
    else { 
        float q = (b > 0) ? 
            -0.5 * (b + std::sqrt(discr)) : 
            -0.5 * (b - std::sqrt(discr)); 
        x0 = q / a; 
        x1 = c / q; 
    } 
 
    return true; 
}

void isect_init(isect &i) {
    i.inear = std::numeric_limits<float>::max(); 
    i.o = NULL; 
}

// returns true if intersection detected
bool ray_sphere_intersect(ray *r, sphere *s, float &inear) {
    float t0, t1; 
    vec4 L = r->src; 
    vec4_sub(&L, &s->pos); 

    double a = vec4_dot(&r->dir, &r->dir);
    double b = 2 * vec4_dot(&r->dir, &L); 
    double c = vec4_dot(&L, &L) - s->radius * s->radius;
    if (!quadratic_solve(a, b, c, t0, t1)) {
        return false; // no intersection 
    }

    if (t0 > t1) std::swap(t0, t1);

    if (t0 < 0) {
        t0 = t1; 
        if (t0 < 0) return false; 
    }

    inear = t0; 
    return true; 
}

/** 
 * Möller–Trumbore intersection algorithm
 * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
 */
bool ray_trig_intersect(ray *r, const trig *trig, float &inear, float &u, float &v) {
    vec4 v0v1 = *trig->v1_trans;
    vec4_sub(&v0v1, trig->v0_trans);
    vec4 v0v2 = *trig->v2_trans;
    vec4_sub(&v0v2, trig->v0_trans);

    vec4 h; 
    vec4_cross(&r->dir, &v0v2, &h);
    double a = vec4_dot(&v0v1, &h);

    if (a > -EPSI_E5 && a < EPSI_E5) {
        // ray is parallel to this triangle 
        return false; 
    }

    double f = 1.0 / a; 

    vec4 s = r->src; 
    vec4_sub(&s, trig->v0_trans);
    u = f * vec4_dot(&s, &h);
    if (u < 0.0 || u > 1.0) {
        return false; 
    }

    vec4 q; 
    vec4_cross(&s, &v0v1, &q);
    v = f * vec4_dot(&r->dir, &q);
    if (v < 0.0 || u + v > 1.0) {
        return false; 
    }

    // now we now there is a non-trivial intersection
    inear = f * vec4_dot(&v0v2, &q); 
    if (inear > EPSI_E5) {
        return true; 
    } else {
        return false; // line but not ray intersection 
    }
} 

bool ray_trig_mesh_intersect(ray *r, trig_mesh *o, float &res, vec2 &uv, uint &trig_index) {
    bool ret = false; 
    for (uint i = 0; i < o->num_polygons; i++) {
        const trig *t = o->tlist[i]; 
        float inear = std::numeric_limits<float>::max(); 
        float u, v; 
        if (ray_trig_intersect(r, t, inear, u, v) && inear < res) {
            res = inear; 
            uv.x = u; 
            uv.y = v; 
            trig_index = i;
            ret = true; 
        }
    }

    return ret; 
}

bool trace(ray *r, std::vector<obj*> &obj_list, isect &res) {
    isect_init(res);

    for (auto it = obj_list.begin(); it != obj_list.end(); it++) {
        float inear = std::numeric_limits<float>::max(); 

        bool ret = false; 
        vec2 uv; 
        uint trig_index; 
        switch ((*it)->type) {
            case SPHERE:
                ret = ray_sphere_intersect(r, (sphere*)(*it), inear); 
                break;
            
            case TRIG_MESH:
                ret = ray_trig_mesh_intersect(r, (trig_mesh*)(*it), inear, uv, trig_index);
                break; 
            
            default:
                break;
        }

        if (ret) {
            if (inear < res.inear) {
                res.o = (*it);
                res.inear = inear; 
                if ((*it)->type == TRIG_MESH) {
                    res.uv = uv; 
                    res.trig_index = trig_index; 
                }
            }
        }
    }

    return (res.o != NULL); 
}

std::default_random_engine generator;
std::uniform_real_distribution<float> dist01;

void random_init(uint seed) {
    generator = std::default_random_engine(seed);
    dist01 = std::uniform_real_distribution<float>(0, 1);
}

vec4 uniform_sample_hemis(const float &r1, const float &r2) {
    // cos(theta) = r1 = y 
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sinTheta = sqrtf(1 - r1 * r1); 
    float phi = 2 * PI * r2; 
    float x = sinTheta * cosf(phi); 
    float z = sinTheta * sinf(phi); 
    return {x, r1, z, 1.0}; 
} 

mat44 create_sample_coord(const vec4 &N) 
{ 
    vec4 Nt, Nb;
    if (std::fabs(N.x) > std::fabs(N.y)) {
        Nt = {N.z, 0, -N.x, std::sqrt(N.x * N.x + N.z * N.z)}; 
        vec4_divide_by_w(&Nt);
    } else {
        Nt = {0, -N.z, N.y, std::sqrt(N.y * N.y + N.z * N.z)}; 
        vec4_divide_by_w(&Nt);
    }
    vec4_cross(&N, &Nt, &Nb); 
    mat44 ret; 
    mat44_init_row(&ret, Nb, N, Nt);
    return ret; 
} 
