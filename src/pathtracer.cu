#include "draw.h"
#include "image.h"
#include "material.h"
#include "maths.h"
#include "object.h"
#include "pathtracer.h"
#include "vector.h"

/**
 * @brief performs both sampling and gamma correction
 * 
 * @param c color
 */
static void gamma_correct(color &c) {
    float scale = 1.0 / SAMPLES_PER_PIX; 
    vec3_mul(&c, scale);
    c.x = std::sqrtf(c.x);
    c.y = std::sqrtf(c.y);
    c.z = std::sqrtf(c.z);
}

/**
 * @brief blue gradient backgrond
 * 
 * @param r ray
 * @return color background color 
 */
static color background_color(ray &r) {
    vec4 unit_direction = r.dir;
    float t = 0.5 * (unit_direction.y + 1.0);
    return (1.f - t) * color { 1.0, 1.0, 1.0 } + t * color{ 0.5, 0.7, 1.0 };
}

static color compute_direct_illum(vec4 pnt, std::vector<obj*> &obj_list, std::vector<light*> &light_list, vec4 i_normal) {
    color ret = {0, 0, 0};
    for (auto it = light_list.begin(); it != light_list.end(); it++) {
        ray shadow_ray; 
        bool shaded = false;
        isect shadow_isect; 
        shadow_ray.src = pnt; 
        switch ((*it)->type) {
            case POINT:
                shadow_ray.dir = ((point_light*)*it)->pos;
                vec4_sub(&shadow_ray.dir, &shadow_ray.src);
                vec4_normalize(&shadow_ray.dir);
                shaded = trace(&shadow_ray, obj_list, shadow_isect); 
                if (shaded) {
                    vec4 dv = ((point_light*)*it)->pos; 
                    vec4_sub(&dv, &shadow_ray.src);
                    float dist = vec4_length(&dv);
                    if (shadow_isect.inear > dist) {
                        shaded = false; // light source is closer 
                    }
                }
                break;
            case DIRECTIONAL:
                shadow_ray.dir = {0, 0, 0, 0};
                vec4_sub(&shadow_ray.dir, &((directional_light*)*it)->dir);
                shaded = trace(&shadow_ray, obj_list, shadow_isect); 
                break;
            case SPOT:
                // Not implemented
                break; 
        } 
        if (!shaded) {
            color contrib = (*it)->col; 
            vec3_mul(&contrib, (*it)->intensity * 
                                std::max(0.0, vec4_dot(&i_normal, &shadow_ray.dir))); 
            vec3_add(&ret, &contrib);
        }       
    }

    return ret; 
}

static color compute_indirect_illum(ray *r, vec4 pnt, std::vector<obj*> &obj_list, std::vector<light*> &light_list, vec4 i_normal, uint depth) {
    color ret = {0, 0, 0}; 
    mat44 sample_to_world = create_sample_coord(i_normal);
    float pdf = 1 / (2 * PI); 

    for (uint i = 0; i < NUM_RAYS; i++) {
        float r1 = random01();
        float r2 = random01();

        vec4 sample = uniform_sample_hemis(r1, r2);
        vec4 sample_trans; 
        mat44_mul(&sample_to_world, &sample, &sample_trans);

        ray ray_w_offset; 
        ray_w_offset.src = pnt;
        ray_w_offset.dir = sample_trans; 
        color contrib = send_light(&ray_w_offset, obj_list, light_list, depth);
        vec3_mul(&contrib, 1 / pdf * r1); // r1 = cos(theta) 
        vec3_add(&ret, &contrib);
    }

    vec3_mul(&ret, 1.0f / NUM_RAYS);

    return ret; 
}

color send_light(ray *r, const std::vector<obj*> &obj_list, const std::vector<light*> &light_list, uint depth) {
    if (depth == 0) { // end recursion 
        return { 0, 0, 0 }; // black 
    }

    color ret = {0, 0, 0}; 
    isect trace_res;
    isect_init(trace_res); 
    
    if (trace(r, obj_list, trace_res)) {
        vec4 i_pnt = r->dir; 
        vec4_mul(&i_pnt, trace_res.inear); 
        vec4_add(&i_pnt, &r->src);

        vec4 i_normal; 
        color i_albedo; 

        illum_type itype = trace_res.o->get_surface(&i_pnt, trace_res.uv, trace_res.trig_index, i_normal, i_albedo);
        vec4 i_pnt_w_offset = i_normal; 
        vec4_mul(&i_pnt_w_offset, NORMAL_BIAS);
        vec4_add(&i_pnt_w_offset, &i_pnt);

        obj *o = trace_res.o; 
        ret = o->obj_mat->get_color(r, i_pnt_w_offset, obj_list, light_list, i_normal, depth);

        return ret; 
    } 

    // return background_color(*r);
    return BACKGROUND;
}

/**
 * @brief paints on to canvas at x=(-1, 1), y = (-1, 1), z = 1
 * 
 * @param obj_list object list 
 * @param light_list light list
 * @param fov field of view, in degrees (suggested = 60)
 * @param width viewport width 
 * @param height viewport height 
 * @param fb framebuffer
 */
__global__ void pathtracer_paint(const std::vector<obj*> obj_list, const std::vector<light*> light_list, float fov, uint width, uint height, framebuffer *fb) {
    omp_set_num_threads(OMP_NUM_THREADS);
    const float scale = tanf(DEG_TO_RAD(fov) / 2);
    const float scale_factor = 1.0; 
    const float aspect = (float)width / (float)height; 
    float start_time = platform_get_time(); 

    int completed = 0;

    int row = threadIdx.x; 
    int col = threadIdx.y; 

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            ray r;
            r.src = { 0, 0, 0 }; // cam at original 
            color sum = { 0, 0, 0 };

            for (int i = 0; i < SAMPLES_PER_PIX; i++) {
                float x = (col + 0.5 - width / 2) / (float)width * aspect * scale * scale_factor;
                float y = (-(row + 0.5 - height / 2) / (float)height) * scale * scale_factor;

                // random sampling 
                x += random01() / (float)width;
                y += random01() / (float)height;

                r.dir = {x, y, 1, 1}; 
                vec4_normalize(&r.dir);
                
                color ret = send_light(&r, obj_list, light_list, 4);
                vec3_add(&sum, &ret);
            }

            gamma_correct(sum);

            draw_point(col, row, sum, fb);
        }
        completed++; 

        float cur_time = platform_get_time();
        printf("%3d %%, %4d seconds elapsed\n", (uint)(completed / (float)height * 100), (uint)(cur_time - start_time)); 
    }
}
