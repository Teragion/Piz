#include <iostream> 
#include <stdio.h>

#include "camera.h"
#include "draw.h"
#include "framebuffer.h"
#include "macros.h"
#include "matrix.h"
#include "object.h"
#include "platform.h"

using namespace std; 

vec4 dv; 
camera cam; 
vector<polyhed*> obj_list; 

void keybd_callback(window_t *window, keycode key, int pressed) {
	switch (key) {
		case KEY_A:
			if (pressed) {
				dv.x = -0.1; 
			} else {
				dv.x = 0; 
			}
			break; 
		case KEY_D:
			if (pressed) {
				dv.x = 0.1; 
			} else {
				dv.x = 0; 
			}
			break; 
		case KEY_W:
			if (pressed) {
				dv.z = 0.1; 
			} else {
				dv.z = 0; 
			}
			break; 
		case KEY_S:
			if (pressed) {
				dv.z = -0.1; 
			} else {
				dv.z = 0; 
			}
			break; 
		default: 
			break; 
	}
}

void main_loop(window_t *window) {
	printf("entered main loop.\n");
	framebuffer *fb = framebuffer_create(1280, 720);

	float prev_time = platform_get_time(); 
	float print_time = prev_time; 
	uint num_frames = 0;

	while (!window_should_close(window)) {
		float cur_time = platform_get_time(); 
		float delta_time = cur_time - prev_time; 
		vec4_add(&cam.pos, &dv);

		framebuffer_ccolor(fb, {0, 0.1, 0.3, 0}); // fill buffer with some color

		// -------------- The Actual Drawing -------------- 
		// delta_time may be used to update
		camera_build_mcam(&cam);
		camera_build_mperspect(&cam);
		camera_build_mscr(&cam);

		for (auto it = obj_list.begin(); it != obj_list.end(); it++) {
			polyhed *o = *it;
			mat44 otrans = o->build_trans_mat();
			o->trans_vlist(&otrans, LOCAL_TO_TRANS); 
			o->trans_vlist(&cam.mcam, TRANS_ONLY); 
			o->trans_vlist(&cam.mperspect, TRANS_ONLY); 
			o->convert_from_homogenous4d();
			o->trans_vlist(&cam.mscr, TRANS_ONLY); 
			unsigned char color[4] = {255, 0, 0, 0};
			draw_polyhed_wireframe(o, color, fb);
		}

		// -------------- The Actual Drawing -------------- 
		window_draw_buffer(window, fb); 
		num_frames++; 
		if (cur_time - print_time >= 1.0) {
			int sum_millis = (int)((cur_time - print_time) * 1000);
			int avg_millis = sum_millis / num_frames;
			printf("fps: %3d, avg: %3d ms\n", num_frames, avg_millis);
			num_frames = 0; 
			print_time = cur_time; 
		}
		prev_time = cur_time; 

		input_poll_events();
	}

	framebuffer_release(fb);
}


int main() {
	platform_init(); 
	printf("init complete.\n");
	window_t *window = window_create("demo", 1280, 720); 
	printf("window created.\n");
	callbacks *callback_list = (callbacks*)malloc(sizeof(callbacks)); 
	callback_list->keybd_callback = keybd_callback; 
	input_set_callbacks(window, *callback_list);
	printf("callbacks binded.\n");
	// -------------- Scene Setup -------------- 
	trig_mesh trig; 
	trig.num_polygons = 0;
	trig.num_vertices = 0;
	trig.pos = {0, 0, 50, 1};
	trig.add_vert({0, 20, 0, 1});
	trig.add_vert({20, 0, 0, 1});
	trig.add_vert({0, 0,  0, 1});

	trig.add_trig(0, 1, 2);
	obj_list.push_back(&trig);

	camera_init(&cam, {0, 0, -50, 1}, {0, 0, 1, 1}, 50, 500, 90, 1280, 720);
	printf("scene created.\n");
	// -------------- Scene Setup -------------- 
	main_loop(window);
	printf("exited mainloop.\n");
	window_destroy(window);
	platform_term(); 
}