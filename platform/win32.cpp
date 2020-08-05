#include <assert.h>
#include <direct.h>
#include <windows.h>
#include <windowsx.h>

#include "platform.h"

struct window {
    HWND handle;
    HDC memory_dc;
    // image* surface;
    /* common data */
    int should_close;
    char keys[KEY_NUM];
    // char buttons[BUTTON_NUM];
    // callbacks callback_list;
    void* userdata;
}; 

static int global_init = 0; 

#ifdef UNICODE
    static const wchar_t* const WINDOW_CLASS_NAME = L"Piz_Demo";
    static const wchar_t* const WINDOW_ENTRY_NAME = L"Piz_Entry";
#else
    static const char* const WINDOW_CLASS_NAME = "Piz_Demo";
    static const char* const WINDOW_ENTRY_NAME = "Piz_Entry";
#endif

void platform_init() {
    assert(global_init == 0); 

}

void platform_term(); 

// window related functions 
static HWND create_window(const char* window_title, int width, int height) {
    DWORD style = WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT rect; 
    HWND handle; 

#ifdef UNICODE 
    wchar_t title[LINE_SIZE];
    mbstowcs(title, window_title, LINE_SIZE); 
#else 
    const char* title = window_title; 
#endif 

    rect.left = 0; 
    rect.top = 0; 
    rect.right = width; 
    rect.bottom = height; 
    AdjustWindowRect(&rect, style, 0); 
    width = rect.right - rect.left; 
    height = rect.bottom - rect.top; 

    handle = CreateWindow(WINDOW_CLASS_NAME, title, style,
        CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        NULL, NULL, GetModuleHandle(NULL), NULL);

    assert(handle != NULL); 
    return handle; 
}

// window controls 
window_t window_create(const char* title, int width, int height) {
    window* ret; 
    HWND handle; 
    HDC memory_dc; 
}

void window_destroy(window_t *window);
void window_set_userdata(window_t *window, void *userdata);
void *window_get_userdata(window_t *window);
void window_draw_buffer(window_t *window, framebuffer *buffer); 

// input events 
void input_poll_events(void);
int input_key_pressed(window_t *window, keycode_t key);
// int input_button_pressed(window_t *window, button_t button);
void input_query_cursor(window_t *window, float *xpos, float *ypos);
// void input_set_callbacks(window_t *window, callbacks_t callbacks);

/* misc platform functions */
float platform_get_time(void);