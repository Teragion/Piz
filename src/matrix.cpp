#include <cmath>

#include "macros.h"
#include "matrix.h"
#include "vector.h"

void mat44_init(mat44* ma, 
                float m00, float m01, float m02, float m03,
                float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23,
                float m30, float m31, float m32, float m33)

{
    ma->M00 = m00; ma->M01 = m01; ma->M02 = m02; ma->M03 = m03;
    ma->M10 = m10; ma->M11 = m11; ma->M12 = m12; ma->M13 = m13;
    ma->M20 = m20; ma->M21 = m21; ma->M22 = m22; ma->M23 = m23;
    ma->M30 = m30; ma->M31 = m31; ma->M32 = m32; ma->M33 = m33;
}

void mat44_init_col(mat44* ma, vec4 c1, vec4 c2, vec4 c3) {
    mat44_init(ma,
               c1.x,    c2.x,   c3.x,   0,
               c1.y,    c2.y,   c3.y,   0,
               c1.z,    c2.z,   c3.z,   0,
               0,       0,      0,      1);
}


void mat44_init_row(mat44* ma, vec4 c1, vec4 c2, vec4 c3) {
    mat44_init(ma,
               c1.x,    c1.y,   c1.z,   0,
               c2.x,    c2.y,   c2.z,   0,
               c3.x,    c3.y,   c3.z,   0,
               0,       0,      0,      1);
}

void mat44_rotate_x(mat44* ma, float theta_x) {
    float s = sin(theta_x);
    float c = cos(theta_x);

    mat44_init(ma, 
               1,   0,  0,  0,
               0,   c, -s,  0,
               0,   s,  c,  0,
               0,   0,  0,  1);
}

void mat44_rotate_y(mat44* ma, float theta_y) {
    float s = sin(theta_y);
    float c = cos(theta_y);

    mat44_init(ma, 
               c,   0,  s,  0,
               0,   1,  0,  0,
              -s,   0,  c,  0,
               0,   0,  0,  1);
}

void mat44_rotate_z(mat44* ma, float theta_z) {
    float s = sin(theta_z);
    float c = cos(theta_z);

    mat44_init(ma, 
               c,  -s,  0,  0,
               s,   c,  0,  0,
               0,   0,  1,  0,
               0,   0,  0,  1);
}

void mat44_rotate_xyz(mat44* ma, float theta_x, float theta_y, float theta_z) {
    mat44 mx, my, mz, mtmp; 
    float sin_theta = 0, cos_theta = 0;
    int rot_seq = 0; // x : 1
                     // y : 2
                     // z : 4

    *ma = IDENTITY44; 

    // determine rotation sequence 
    if (abs(theta_x) > EPSI_E5) 
        rot_seq = rot_seq | 1;

    if (abs(theta_y) > EPSI_E5)
        rot_seq = rot_seq | 2; 
    
    if (abs(theta_z) > EPSI_E5)
        rot_seq = rot_seq | 4;

    
}

void mat44_mul(mat44* ma, mat44* mb, mat44* mres) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            double sum = 0; 
            for (int i = 0; i < 4; i++)
                sum += *(*ma)(r, i) * *(*mb)(i, c);

            *(*mres)(r, c) = (float)sum;
        }
    }
}

// currently using row vectors 
void mat44_mul(mat44* ma, vec4* v, vec4* vres) {
    for (int c = 0; c < 4; c++) {
        double sum = 0; 
        for (int i = 0; i < 4; i++) {
            sum += *(*ma)(i, c) * v->M[i];
        }
        vres->M[c] = (float)sum;
    }
}
