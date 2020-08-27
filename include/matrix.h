#ifndef MATRIX_H
#define MATRIX_H

#include "macros.h"
#include "vector.h"

// TODO: improve the assignment mode 

struct mat33 {
    union {
        float M[9];
        struct
        {
            float M00, M01, M02;
            float M10, M11, M12;
            float M20, M21, M22;
        }; // explicit names
    };

    __host__ __device__ inline float* operator() (int row, int col) {
        return &M[row * 3 + col]; 
    }
};

struct mat43 {
    union {
        float M[12];
        struct
        {
            float M00, M01, M02;
            float M10, M11, M12;
            float M20, M21, M22;
            float M30, M31, M32;
        }; // explicit names
    };

    __host__ __device__ inline float* operator() (int row, int col) {
        return &M[row * 3 + col]; 
    }
};

struct mat44 {
    union {
        float M[16];
        struct
        {
            float M00, M01, M02, M03;
            float M10, M11, M12, M13;
            float M20, M21, M22, M23;
            float M30, M31, M32, M33;
        }; // explicit names
    };

    __host__ __device__ inline float* operator() (int row, int col) {
        return &M[row * 4 + col]; 
    }
};

// constants 
const mat44 IDENTITY44 = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};


__host__ __device__ void mat44_init(mat44* ma, 
                float m00, float m01, float m02, float m03,
                float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23,
                float m30, float m31, float m32, float m33);

__host__ __device__ void mat44_init_col(mat44* ma, vec4 c1, vec4 c2, vec4 c3); 
__host__ __device__ void mat44_init_row(mat44* ma, vec4 r1, vec4 r2, vec4 r3); 
__host__ __device__ void mat44_rotate_x(mat44* ma, float theta_x);
__host__ __device__ void mat44_rotate_y(mat44* ma, float theta_y);
__host__ __device__ void mat44_rotate_z(mat44* ma, float theta_z);
__host__ __device__ void mat44_rotate_xyz(mat44* ma, float theta_x, float theta_y, float theta_z);

__host__ __device__ void mat44_mul(mat44* ma, mat44* mb, mat44* mres); 

__host__ __device__ void mat44_mul(mat44* ma, vec4* v, vec4* vres);


#endif
