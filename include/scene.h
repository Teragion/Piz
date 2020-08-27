#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "object.h"
#include "light.h"

struct scene {
    __host__ __device__ virtual void init_scene(std::vector<obj*>& obj_list, std::vector<light*>& light_list) = 0; 
};

struct scene_my_tracer : scene {
    __host__ __device__ virtual void init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list);
};

struct scene_cornell_box : scene {
    __host__ __device__ virtual void init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list);
}; 

#endif
