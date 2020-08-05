#ifndef POLYGON_H
#define POLYGON_H

#include <vector>

#include "macros.h"

struct polygon {
    uint num_vertices; 
    std::vector<uint> vlist; 
};

#endif 
