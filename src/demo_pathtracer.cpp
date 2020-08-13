#include "stdio.h"

#include "pathtracer.h"
#include "platform.h"

std::vector<obj*> obj_list; 
std::vector<light*> light_list; 

int main() {
	platform_init(); 
    random_init();
	printf("init complete.\n");
	// -------------- Scene Setup -------------- 
	sphere s1; 
	s1.type = SPHERE;
	s1.itype = DIFFUSE;
    s1.pos = {0, 0, 50, 1};
    s1.radius = 40.0;
    s1.albedo = {0.18, 0.18, 0.18};
	obj_list.push_back(&s1);

	printf("scene created.\n");
	// -------------- Scene Setup -------------- 
	framebuffer *fb = framebuffer_create(640, 360);
    pathtracer_paint(obj_list, light_list, 90, 640, 360, fb);
	platform_term(); 
}