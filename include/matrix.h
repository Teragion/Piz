#ifndef MATRIX_H
#define MATRIX_H

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

    inline float* operator() (int row, int col) {
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

    inline float* operator() (int row, int col) {
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

    inline float* operator() (int row, int col) {
        return &M[row * 4 + col]; 
    }
};

void mat44_init(mat44* ma, 
                float m00, float m01, float m02, float m03,
                float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23,
                float m30, float m31, float m32, float m33);

void mat44_init_col(mat44* ma, vec4 c1, vec4 c2, vec4 c3); 
void mat44_init_row(mat44* ma, vec4 r1, vec4 r2, vec4 r3); 

// Assumes last col of matrices are all {0, 0, 0, 1}
void mat44_mul(mat44* ma, mat44* mb, mat44* mres); 

#endif
