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
typedef vec3 color;

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

// TODO: make some pointers const 

__host__ __device__ inline void vec3_init(vec3* v, float x, float y, float z) {
    v->x = x; v->y = y; v->z = z;
}

__host__ __device__ inline void vec3_reset(vec3* v) {
    vec3_init(v, 0, 0, 0);
}

__host__ __device__ inline void vec3_add(vec3* a, const vec3* b) {
    a->x += b->x; a->y += b->y; a->z += b->z; 
}

__host__ __device__ inline vec3 operator+(const vec3 &a, const vec3 &b) {
    vec3 ret = a; 
    vec3_add(&ret, &b);
    return ret; 
}

__host__ __device__ inline void vec3_mul(vec3* a, vec3* b) {
    a->x *= b->x; a->y *= b->y; a->z *= b->z; 
}

__host__ __device__ inline void vec3_mul(vec3* a, float b) {
    a->x *= b; a->y *= b; a->z *= b;  
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    vec3 ret = v;
    vec3_mul(&ret, t); 
    return ret;
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v; 
}

__host__ __device__ void vec3_normalize(vec3* v);

__host__ __device__ inline void vec4_init(vec4* v, float x, float y, float z, float w = 1.0) {
    v->x = x; v->y = y; v->z = z; v->w = w; 
}

__host__ __device__ inline void vec4_reset(vec4* v) {
    vec4_init(v, 0, 0, 0, 0);
}

__host__ __device__ inline void vec4_add(vec4* a, vec4* b) {
    a->x += b->x; a->y += b->y; a->z += b->z; a->w = 1;  
}

__host__ __device__ inline void vec4_sub(vec4* a, vec4* b) {
    a->x -= b->x; a->y -= b->y; a->z -= b->z; a->w = 1;  
}

__host__ __device__ inline void vec4_mul(vec4* a, float b) {
    a->x *= b; a->y *= b; a->z *= b; a->w = 1;  
}

// normalize first 3 elements of v 
__host__ __device__ void vec4_normalize(vec4* v);

__host__ __device__ double vec4_length(vec4* v);

__host__ __device__ void vec4_divide_by_w(vec4* v);

// res = a x b (as vec3)
__host__ __device__ void vec4_cross(const vec4* a, const vec4* b, vec4* res);
__host__ __device__ double vec4_dot(vec4* a, vec4* b);


#endif 
