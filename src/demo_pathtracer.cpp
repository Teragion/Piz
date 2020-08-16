#include "stdio.h"

#include "material.h"
#include "pathtracer.h"
#include "platform.h"
#include "scene.h"

std::vector<obj*> obj_list; 
std::vector<light*> light_list; 

#define WINDOW_WIDTH	720
#define WINDOW_HEIGHT 	720

int main() {
	platform_init(); 
    random_init();
	printf("init complete.\n");

	// scene_my_tracer scene; 
	scene_cornell_box scene;
	scene.init_scene(obj_list, light_list); 
	printf("scene created.\n");

	framebuffer *fb = framebuffer_create(WINDOW_WIDTH, WINDOW_HEIGHT);
    pathtracer_paint(obj_list, light_list, 67, WINDOW_WIDTH, WINDOW_HEIGHT, fb);

    image *img = image_create(WINDOW_WIDTH, WINDOW_HEIGHT, 4, FORMAT_LDR);
    blit_rgba(fb, img);
    image_flip_v(img);

    image_save(img, "result.tga");

	platform_term(); 
	
	return 0; 
}