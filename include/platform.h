#ifndef PLATFORM_H
#define PLATFORM_H

#include "framebuffer.h"

typedef struct window window_t; 
enum keycode {KEY_A, KEY_D, KEY_S, KEY_W, KEY_SPACE, KEY_NUM};
enum mbutton {MOUSE_L, MOUSE_R, MOUSE_NUM};

typedef struct callbacks_t {
    void (*keybd_callback)(window_t *window, keycode key, int pressed);
    void (*mbutton_callback)(window_t *window, mbutton button, int pressed);
    void (*scroll_callback)(window_t *window, float offset);
} callbacks;

// Construction 
void platform_init(); 
void platform_term(); 

// window controls 
window_t* window_create(const char *title, int width, int height); 
void window_destroy(window_t *window);
// void window_set_userdata(window_t *window, void *userdata);
// void *window_get_userdata(window_t *window);
void window_draw_buffer(window_t *window, framebuffer *buffer); 

int window_should_close(window_t *window);

// input events 
void input_poll_events();
int input_key_pressed(window_t *window, keycode key);
// int input_button_pressed(window_t *window, button_t button);
void input_query_cursor(window_t *window, float *xpos, float *ypos);
void input_set_callbacks(window_t *window, callbacks callback_list);

// misc platform functions
float platform_get_time();

#endif
