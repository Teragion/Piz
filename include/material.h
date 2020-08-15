#ifndef MATERIAL_H
#define MATERIAL_H

#include "light.h"
#include "object.h"
#include "vector.h"

struct material {
    color albedo; 

    material(color& _albedo);

    /**
     * @return true on default 
    */
    virtual bool is_transparent();

    virtual ray hit(vec4 i_pnt, vec4 dir_in, vec4 i_normal, color& attenuation);

    virtual color compute_direct_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal);

    virtual color compute_indirect_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth);

    virtual color get_color(ray* r, vec4 i_pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth);
};

struct material_transparent : material {

};

struct material_diffuse : material {
    material_diffuse(color& _albedo) :
        material(_albedo) {}

    virtual bool is_transparent() {
        return false; 
    }

    /**
     * @brief returns sample of diffused ray in sample coordinate 
     * @param i_pnt intersect point
     * @param dir_in ray direction
     * @param i_normal surface normal
     * @param attenuation attenuation
     * @return direction of ray in sample coordinate
    */
    virtual ray hit(vec4 i_pnt, vec4 dir_in, vec4 i_normal, color &attenuation);

    virtual color compute_direct_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal);

    virtual color compute_indirect_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth); 
};

struct material_specular : material {
    color attenuation; // perfect mirror : attenuation = 0 
    float fuzziness; // (0, 1)

    material_specular(color &_albedo, color &_attenuation, float _fuzziness) :
        material(_albedo), 
        attenuation(_attenuation), 
        fuzziness(_fuzziness) {}

    virtual bool is_transparent() {
        return false; 
    }

    /**
     * @brief returns sample of reflected ray in world coordinate 
     * @param i_pnt intersect point
     * @param dir_in ray direction
     * @param i_normal surface normal
     * @param attenuation attenuation
     * @return direction of ray in world coordinate
    */
    virtual ray hit(vec4 i_pnt, vec4 dir_in, vec4 i_normal, color &attenuation); 

    virtual color compute_direct_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal);

    virtual color compute_indirect_illum(ray* r, vec4 pnt, const std::vector<obj*>& obj_list, const std::vector<light*>& light_list, vec4 i_normal, uint depth);
};

#endif 
