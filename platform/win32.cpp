#include <assert.h>
#include <direct.h>
#include <windows.h>
#include <windowsx.h>

#include "image.h"
#include "platform.h"

struct window {
    HWND handle;
    HDC memory_dc;
    image* surface;
    /* common data */
    int should_close;
    char keys[KEY_NUM];
    char mbuttons[MOUSE_NUM];
    callbacks callback_list;
    void* userdata;
}; 

static int global_init = 0; 
static double initial_time; 

static double get_native_time();

/*
 * for virtual-key codes, see
 * https://docs.microsoft.com/en-us/windows/desktop/inputdev/virtual-key-codes
 */
static void handle_key_message(window_t *window, WPARAM virtual_key, char pressed) {
    keycode key; 
    switch (virtual_key) {
        case 'A':       key = KEY_A;        break;
        case 'D':       key = KEY_D;        break;
        case 'S':       key = KEY_S;        break;
        case 'W':       key = KEY_W;        break;
        case VK_SPACE:  key = KEY_SPACE;    break;
        default:        key = KEY_NUM;      break;
    }

    if (key < KEY_NUM) {
        window->keys[key] = pressed; 
        if (window->callback_list.keybd_callback) {
            window->callback_list.keybd_callback(window, key, pressed);
        }
    }
}

#ifdef UNICODE
    static const wchar_t* const WINDOW_CLASS_NAME = L"Piz_Demo";
    static const wchar_t* const WINDOW_ENTRY_NAME = L"Piz_Entry";
#else
    static const char* const WINDOW_CLASS_NAME = "Piz_Demo";
    static const char* const WINDOW_ENTRY_NAME = "Piz_Entry";
#endif

// callback related 
static LRESULT CALLBACK process_message(HWND hwnd, UINT uMsg, 
                                        WPARAM wParam, LPARAM lParam) {
    window_t *window = (window_t*)GetProp(hwnd, WINDOW_ENTRY_NAME); 
    if (window == NULL) {
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    } else if (uMsg == WM_CLOSE) {
        window->should_close = 1;
        return 0;
    } else if (uMsg == WM_KEYDOWN) {
        handle_key_message(window, wParam, 1);
        return 0; 
    } else if (uMsg == WM_KEYUP) {
        handle_key_message(window, wParam, 0);
        return 0;
    } else if (uMsg == WM_LBUTTONDOWN) {
        // add mouse handlings 
        return 0;
    } else if (uMsg == WM_RBUTTONDOWN) {
        // add mouse handlings 
        return 0;
    } else if (uMsg == WM_LBUTTONUP) {
        // add mouse handlings 
        return 0;
    } else if (uMsg == WM_RBUTTONUP) {
        // add mouse handlings 
        return 0;
    } else if (uMsg == WM_MOUSEWHEEL) {
        // add mouse handlings 
        return 0;
    } else {
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

// windows class related 
static void register_class() {
    ATOM class_atom; 
    // TODO: maybe use WNDCLASSEX? 
    WNDCLASS window_class; 
    window_class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
    window_class.lpfnWndProc = process_message; 
    window_class.cbClsExtra = 0;
    window_class.cbWndExtra = 0;
    window_class.hInstance = GetModuleHandle(NULL); 
    window_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    window_class.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    window_class.lpszMenuName = NULL; 
    window_class.lpszClassName = WINDOW_CLASS_NAME; 
    class_atom = RegisterClass(&window_class);  
    assert(class_atom != 0); 
    // unused variable 
    (void)class_atom;
}

static void unregister_class() {
    UnregisterClass(WINDOW_CLASS_NAME, GetModuleHandle(NULL));
}

void platform_init() {
    assert(global_init == 0); 
    register_class(); 
    // TODO: do we need initialize_path here?
    initial_time = get_native_time();
    global_init = 1; 

}

void platform_term() {
    assert(global_init == 1); 
    unregister_class();
    global_init = 0;  
}

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
/*
 * See https://docs.microsoft.com/en-us/windows/desktop/gdi/memory-device-contexts
 */
static void create_surface(HWND handle, int width, int height,
                           image **out_surface, HDC *out_memory_dc) {
    BITMAPINFOHEADER bi_header; 
    HBITMAP dib_bitmap; 
    HBITMAP old_bitmap; 
    HDC window_dc; 
    HDC memory_dc; 
    image *surface; 

    surface = image_create(width, height, 4, FORMAT_LDR); 
    free(surface->ldr_buffer); 
    surface->ldr_buffer = NULL; 

    window_dc = GetDC(handle); 
    memory_dc = CreateCompatibleDC(window_dc); 
    ReleaseDC(handle, window_dc); 

    memset(&bi_header, 0, sizeof(BITMAPINFOHEADER));
    bi_header.biSize = sizeof(BITMAPINFOHEADER); 
    bi_header.biWidth = width; 
    bi_header.biHeight = -height;
    bi_header.biPlanes = 1;
    bi_header.biBitCount = 32; 
    bi_header.biCompression = BI_RGB; 
    dib_bitmap = CreateDIBSection(memory_dc, (BITMAPINFO*)&bi_header, 
                                  DIB_RGB_COLORS, (void**)&surface->ldr_buffer,
                                  NULL, 0);
    assert(dib_bitmap != NULL); 
    old_bitmap = (HBITMAP)SelectObject(memory_dc, dib_bitmap);
    DeleteObject(old_bitmap); 

    *out_surface = surface; 
    *out_memory_dc = memory_dc;  
} 

// window controls 
window_t* window_create(const char* title, int width, int height) {
    assert(global_init && width > 0 && height > 0); 

    window_t* ret; 
    HWND handle; 
    image *surface; 
    HDC memory_dc; 

    handle = create_window(title, width, height); 
    create_surface(handle, width, height, &surface, &memory_dc); 

    ret = (window_t*)malloc(sizeof(window_t));
    memset(ret, 0, sizeof(window_t));
    ret->handle = handle; 
    ret->memory_dc = memory_dc; 
    ret->surface = surface; 

    SetProp(handle, WINDOW_ENTRY_NAME, ret); 
    ShowWindow(handle, SW_SHOW); 
    return ret; 
}

void window_destroy(window_t *window) {
    ShowWindow(window->handle, SW_HIDE); 
    RemoveProp(window->handle, WINDOW_ENTRY_NAME);

    DeleteDC(window->memory_dc); 
    DestroyWindow(window->handle); 

    window->surface->ldr_buffer = NULL; 
    image_release(window->surface);
    free(window);
}

// void window_set_userdata(window_t *window, void *userdata);
// void *window_get_userdata(window_t *window);

// buffer/surface related
static void present_surface(window_t *window) {
    HDC window_dc = GetDC(window->handle);
    HDC memory_dc = window->memory_dc; 
    image *surface = window->surface;
    int width = surface->width; 
    int height = surface->height; 
    BitBlt(window_dc, 0, 0, width, height, memory_dc, 0, 0, SRCCOPY); 
    ReleaseDC(window->handle, window_dc); 
}

void window_draw_buffer(window_t *window, framebuffer *buffer) {
    blit_bgr(buffer, window->surface); // windows rgb sequence? 
    present_surface(window);
} 

// input events 
void input_poll_events() {
    MSG message; 
    while (PeekMessage(&message, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }
}

// keyboard related 

int input_key_pressed(window_t *window, keycode key);
// int input_button_pressed(window_t *window, button_t button);
void input_query_cursor(window_t *window, float *xpos, float *ypos);
void input_set_callbacks(window_t *window, callbacks callback_list) {
    window->callback_list = callback_list;
}

// private data 
int window_should_close(window_t *window) {
    return window->should_close; 
}

/* misc platform functions */
static double get_native_time() {
    static double period = -1;
    LARGE_INTEGER counter;
    if (period < 0) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        period = 1 / (double)frequency.QuadPart;
    }
    QueryPerformanceCounter(&counter);
    return counter.QuadPart * period;
}

float platform_get_time() {
    return (float)(get_native_time() - initial_time); 
}
