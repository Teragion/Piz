#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "light.h"
#include "macros.h"
#include "maths.h"
#include "object.h"
#include "platform.h"
#include "vector.h"

color send_light(ray *r, const std::vector<obj*> &obj_list, const std::vector<light*> &light_list, uint depth);

void pathtracer_paint(const std::vector<obj*> obj_list, const std::vector<light*> light_list, float fov, uint width, uint height, framebuffer *fb); 

#endif 
