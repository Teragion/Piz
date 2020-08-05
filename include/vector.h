#ifndef VECTOR_H
#define VECTOR_H

struct vec2 {
    union 
    {
        float M[2];
        struct {
            float x; 
            float y; 
        };
    };
};

typedef vec2 point2; 

struct vec3 {
    union 
    {
        float M[3];
        struct {
            float x; 
            float y;
            float z; 
        };
    };
};

typedef vec3 point3;

struct vec4 {
    union 
    {
        float M[4];
        struct {
            float x; 
            float y;
            float z; 
            float w; 
        };
    };
};

typedef vec4 point4; 

inline void vec3_init(vec3* v, float x, float y, float z) {
    v->x = x; v->y = y; v->z = z;
}

inline void vec3_reset(vec3* v) {
    vec3_init(v, 0, 0, 0);
}

inline void vec3_add(vec3* a, vec3* b) {
    a->x += b->x; a->y += b->y; a->z += b->z; 
}

void vec3_normalize(vec3* v);

inline void vec4_init(vec4* v, float x, float y, float z, float w = 1.0) {
    v->x = x; v->y = y; v->z = z; v->w = w; 
}

inline void vec4_reset(vec4* v) {
    vec4_init(v, 0, 0, 0, 0);
}

inline void vec4_add(vec4* a, vec4* b) {
    a->x += b->x; a->y += b->y; a->z += b->z; a->w = 1;  
}

inline void vec4_sub(vec4* a, vec4* b) {
    a->x -= b->x; a->y -= b->y; a->z -= b->z; a->w = 1;  
}

// normalize first 3 elements of v 
void vec4_normalize(vec4* v);

// res = a x b (as vec3)
void vec4_cross(vec4* a, vec4* b, vec4* res);


#endif 
