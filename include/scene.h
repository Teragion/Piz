#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "object.h"
#include "light.h"

struct scene {
    virtual void init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list) { 
        return; 
    }
};

struct scene_my_tracer : scene {
    virtual void init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list); 
};

struct scene_cornell_box : scene {
    virtual void init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list); 
}; 

#endif
