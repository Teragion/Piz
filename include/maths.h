#ifndef MATHS_H
#define MATHS_H

// Type definitions 

// vector/point

// matrix

// quaternion

// curve

// Functions 

// TODO: use sin/cos lookup tables to speedup computation. Actually, 
//       check if it actually increases performance 

// converts float in range (0, 1) to unsigned char 
inline unsigned char float_to_uchar(float x) {
    return (unsigned char)x * 255; 
}


#endif
