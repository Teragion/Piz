#include "stdio.h"

#include "pathtracer.h"
#include "platform.h"

std::vector<obj*> obj_list; 
std::vector<light*> light_list; 

#define WINDOW_WIDTH	1280
#define WINDOW_HEIGHT	720

int main() {
	platform_init(); 
    random_init();
	printf("init complete.\n");
	// -------------- Scene Setup -------------- 
	sphere s1; 
	s1.type = SPHERE;
	s1.itype = DIFFUSE;
    s1.pos = {-10, -5, 100, 1};
    s1.radius = 5.0;
    s1.albedo = {0.92, 0.12, 0.12};
	obj_list.push_back(&s1);

    sphere s2; 
	s2.type = SPHERE;
	s2.itype = DIFFUSE;
    s2.pos = {10, -3, 70, 1};
    s2.radius = 7.0;
    s2.albedo = {0.12, 0.92, 0.12};
	obj_list.push_back(&s2);

	trig_mesh m1;
	m1.type = TRIG_MESH;
	m1.itype = DIFFUSE;
	m1.num_vertices = 0;
	m1.num_polygons = 0;
	m1.add_vert({ -30, -10, 150, 1 });
	m1.add_vert({ 30, -10, 150, 1 });
	m1.add_vert({ -30, -10, 50, 1 });
	m1.add_vert({ 30, -10, 50, 1 });
	m1.add_trig(0, 1, 2);
	m1.add_trig(1, 3, 2);
	//m1.add_trig(2, 0, 1);
	//m1.add_trig(3, 1, 2);
	m1.albedo = { 0.6, 0.6, 0.6 };
	obj_list.push_back(&m1);

	trig_mesh m2;
	m2.type = TRIG_MESH;
	m2.itype = MIRROR;
	m2.num_vertices = 0;
	m2.num_polygons = 0;
	m2.add_vert({ -30, 40, 150, 1 });
	m2.add_vert({ 30, 40, 150, 1 });
	m2.add_vert({ -30, -10, 150, 1 });
	m2.add_vert({ 30, -10, 150, 1 });
	m2.add_trig(0, 1, 2);
	m2.add_trig(1, 3, 2);
	// m2.albedo = { 0.3, 0.3, 0.3 };
	// obj_list.push_back(&m2);

	point_light l1; 
	l1.type = POINT; 
	l1.col = { 1, 1, 1 }; 
	l1.intensity = 1.0; 
	l1.pos = { 0, 50, 130, 1 };
	// vec4_normalize(&l1.dir); 
	// light_list.push_back(&l1);

	directional_light l2;
	l2.type = DIRECTIONAL;
	l2.col = { 1, 1, 1 };
	l2.intensity = 2.0;
	l2.dir = { 0, -1, 0, 1 };
	vec4_normalize(&l2.dir); 
	// light_list.push_back(&l2);

	printf("scene created.\n");
	// -------------- Scene Setup -------------- 
	framebuffer *fb = framebuffer_create(WINDOW_WIDTH, WINDOW_HEIGHT);
    pathtracer_paint(obj_list, light_list, 60, WINDOW_WIDTH, WINDOW_HEIGHT, fb);
	platform_term(); 
}