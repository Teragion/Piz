#include "material.h"
#include "maths.h"
#include "pathtracer.h"

material::material(color& _albedo) :
    albedo(_albedo) {}

bool material::is_transparent() {
    return true;
}

color material::compute_direct_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal) {
    return { 0, 0, 0 };
}

color material::compute_indirect_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth) {
    return { 0, 0, 0 };
}

color material::get_color(ray* r, vec4 i_pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth) {
    color direct_illum = compute_direct_illum(r, i_pnt, obj_list, light_list, i_normal);
    color indirect_illum = compute_indirect_illum(r, i_pnt, obj_list, light_list, i_normal, depth);

    color ret = direct_illum;
    vec3_add(&ret, &indirect_illum);
    vec3_mul(&ret, &albedo);

    return ret;
}

ray material::hit(vec4 i_pnt, vec4 dir_in, vec4 i_normal, color& attenuation) {
    return ray{};
}

ray material_diffuse::hit(vec4 i_pnt, vec4 dir_in, vec4 i_normal, color &attenuation) {
    ray ret; 

    float r1 = random01();
    float r2 = random01();

    vec4 sample = uniform_sample_hemis(r1, r2);

    // mat44_mul(&sample_to_world, &sample, &sample_trans); 

    ret.src = i_pnt; 
    ret.dir = sample; 

    attenuation = { 1, 1, 1 }; 
    vec3_mul(&attenuation, r1); // multiply by cos(theta)

    return ret;
}

color material_diffuse::compute_direct_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal) {
    color ret = { 0, 0, 0 };
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
            shadow_ray.dir = { 0, 0, 0, 0 };
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

    vec3_mul(&ret, 1.0 / PI); // BRDF

    return ret;
}

color material_diffuse::compute_indirect_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth) {
    if (depth == 1) { // avoid meaningless calls
        return { 0, 0, 0 }; // black 
    }

    color ret = { 0, 0, 0 };
    mat44 sample_to_world = create_sample_coord(i_normal);
    float pdf = 1 / (2 * PI);

    for (uint i = 0; i < NUM_RAYS; i++) {
        color attenuation; 

        ray ray_w_offset = hit(pnt, r->dir, i_normal, attenuation);
        vec4 tmp;
        mat44_mul(&sample_to_world, &ray_w_offset.dir, &tmp);
        ray_w_offset.dir = tmp;
        color contrib = send_light(&ray_w_offset, obj_list, light_list, depth - 1);
        vec3_mul(&contrib, &attenuation);
        vec3_mul(&contrib, 1 / pdf); 
        vec3_add(&ret, &contrib);
    }

    vec3_mul(&ret, 1.0f / NUM_RAYS);
    vec3_mul(&ret, 1.0 / PI); // BRDF

    return ret;
}

ray material_specular::hit(vec4 i_pnt, vec4 dir_in, vec4 i_normal, color &attenuation) {
    ray ret; 
    ret.src = i_pnt; 
    
    vec4 tmp = i_normal; 
    vec4_mul(&tmp, 2 * vec4_dot(&dir_in, &i_normal));
    ret.dir = dir_in;
    vec4_sub(&ret.dir, &tmp);

    // add fuzziness 
    float r1 = random02Pi(); 
    float r2 = random01(); 
    vec4 fuzz = uniform_sample_sphere(r1, r2);
    vec4_mul(&fuzz, fuzziness);
    vec4_add(&ret.dir, &fuzz); 
    vec4_normalize(&ret.dir);

    attenuation = this->attenuation; 
    vec3_mul(&attenuation, cosf(r1)); 

    // compute pdf 
    float pdf = std::abs(cosf(r1)) / 4;
    vec3_mul(&attenuation, pdf);

    return ret; 
}

color material_specular::compute_direct_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal) {
    color ret = { 0, 0, 0 };
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
            shadow_ray.dir = { 0, 0, 0, 0 };
            vec4_sub(&shadow_ray.dir, &((directional_light*)*it)->dir);
            shaded = trace(&shadow_ray, obj_list, shadow_isect);
            break;
        case SPOT:
            // Not implemented
            break;
        }
        if (!shaded) {
            color contrib = (*it)->col;

            float bound = fuzziness; 
            bound = std::sqrtf(1.0 - bound * bound); // turn into cosine 
            float cosTheta = vec4_dot(&i_normal, &shadow_ray.dir);
            float diff = (cosTheta - bound) / fuzziness; 

            vec3_mul(&contrib, (*it)->intensity * std::asin(diff));
            vec3_add(&ret, &contrib);
        }
    }

    clamp(ret); // do we need this? 

    return ret;
}

color material_specular::compute_indirect_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth) {
    if (depth == 1) { // avoid meaningless calls 
        return { 0, 0, 0 }; // black 
    }

    color attenuation; 
    color ret = { 0, 0, 0 };

    for (uint i = 0; i < NUM_RAYS; i++) {
        ray reflected_ray = hit(pnt, r->dir, i_normal, attenuation);
        color contrib = send_light(&reflected_ray, obj_list, light_list, depth - 1);
        vec3_mul(&contrib, &attenuation); // pdf is already applied 
        vec3_add(&ret, &contrib);
    }

    vec3_mul(&ret, 1.0f / NUM_RAYS);
    return ret; 
}

color material_emissive::get_color(ray* r, vec4 i_pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth) {
    return col * intensity; 
}
