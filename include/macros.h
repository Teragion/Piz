#ifndef MACROS_H
#define MACROS_H

typedef unsigned int uint; 

// Expression comparison 
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

// Swap variables 
#define SWAP(a,b,t) {t=a; a=b; b=t;}

// Math

// epsilon
#define EPSI_E5    ((float)(1E-5))

// pi defines
#define PI         ((float)3.141592654f)
#define PI2        ((float)6.283185307f)
#define PI_DIV_2   ((float)1.570796327f)
#define PI_DIV_4   ((float)0.785398163f) 
#define PI_INV     ((float)0.318309886f) 

/// conversions 
#define DEG_TO_RAD(ang) ((ang)*PI/180.0)
#define RAD_TO_DEG(rad) ((rad)*180.0/PI)

// Other constants 
#define LINE_SIZE	256
#define PATH_SIZE	256

#endif
