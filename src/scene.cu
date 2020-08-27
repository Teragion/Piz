#include "material.h"
#include "scene.h"

static vec4 cornell_offset(vec4 v) {
    static const int cam_offset_x = -278; 
    static const int cam_offset_y = -273; 
    static const int cam_offset_z = 800;

    v.x += cam_offset_x; 
    v.y += cam_offset_y; 
    v.z += cam_offset_z; 
    v.w = 1;

    return v; 
}

static void cornell_offset(polyhed &p) {
    for (int i = 0; i < p.num_vertices; i++) {
        p.vlist_local[i] = cornell_offset(p.vlist_local[i]);
        p.vlist_trans[i] = p.vlist_local[i];
    }
}

/**
 * @param m 
 * @param v0 top left 
 * @param v1 top right 
 * @param v2 bot left 
 * @param v3 bot right 
 */
static void add_rect(trig_mesh &m, vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    int off = m.num_vertices; 

    m.add_vert(v0);
    m.add_vert(v1);
    m.add_vert(v2);
    m.add_vert(v3);

    m.add_rect(off, off + 1, off + 2, off + 3); 
}

void scene_my_tracer::init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list) {
    sphere *s1 = new sphere(); 
	s1->type = SPHERE;
	s1->itype = DIFFUSE;
    s1->pos = {-10, -5, 100, 1};
    s1->radius = 5.0;
	color red = { 0.92, 0.12, 0.12 };
	s1->obj_mat = std::make_shared<material_diffuse>(red);
	obj_list.push_back(s1);

    sphere *s2 = new sphere();
	s2->type = SPHERE;
	s2->itype = DIFFUSE;
    s2->pos = {10, -3, 70, 1};
    s2->radius = 7.0;
    color green = {0.12, 0.92, 0.12};
	s2->obj_mat = std::make_shared<material_diffuse>(green);
	obj_list.push_back(s2);

	trig_mesh *m1 = new trig_mesh(); 
    m1->init_trig();
	m1->type = TRIG_MESH;
	m1->itype = DIFFUSE;
	m1->num_vertices = 0;
	m1->num_polygons = 0;
	m1->add_vert({ -30, -10, 150, 1 });
	m1->add_vert({ 30, -10, 150, 1 });
	m1->add_vert({ -30, -10, 50, 1 });
	m1->add_vert({ 30, -10, 50, 1 });
	m1->add_trig(0, 1, 2);
	m1->add_trig(1, 3, 2);
	color grey = { 0.6, 0.6, 0.6 };
	m1->obj_mat = std::make_shared<material_diffuse>(grey);
	obj_list.push_back(m1);

	trig_mesh *m2 = new trig_mesh();
    m2->init_trig();
	m2->type = TRIG_MESH;
	m2->itype = MIRROR;
	m2->num_vertices = 0;
	m2->num_polygons = 0;
	m2->add_vert({ -30, 40, 150, 1 });
	m2->add_vert({ 30, 40, 150, 1 });
	m2->add_vert({ -30, -10, 150, 1 });
	m2->add_vert({ 30, -10, 150, 1 });
	m2->add_trig(0, 1, 2);
	m2->add_trig(1, 3, 2);
}

void scene_cornell_box::init_scene(std::vector<obj*> &obj_list, std::vector<light*> &light_list) {
    color white = {0.8, 0.7, 0.48};
    color green = {0.1, 0.7, 0.1};
    color red = {0.7, 0.1, 0.1};

    trig_mesh *floor = new trig_mesh(); 
    add_rect(*floor, cornell_offset({0, 0, 559.2, 1}),
                     cornell_offset({549.6, 0, 559.2, 1}), 
                     cornell_offset({0, 0, 0, 1}),
                     cornell_offset({552.8, 0, 0, 1}));
    floor->obj_mat = std::make_shared<material_diffuse>(white);
    obj_list.push_back(floor);

    point_light *light = new point_light(); 
    light->pos = {270, 548, 270, 1}; 
    light->pos = cornell_offset(light->pos);
    light->col = white; 
    light->intensity = 0.8; 
    // light_list.push_back(light);

    trig_mesh *iamlight = new trig_mesh(); 
    add_rect(*iamlight, cornell_offset({208, 548.0, 227, 1}),
                        cornell_offset({343, 548.0, 227, 1}), 
                        cornell_offset({208, 548.0, 332, 1}),
                        cornell_offset({343, 548.0, 332, 1}));
    iamlight->obj_mat = std::make_shared<material_emissive>(white, 20.0);
    obj_list.push_back(iamlight);

    trig_mesh *ceiling = new trig_mesh(); 
    add_rect(*ceiling, cornell_offset({0, 548.8, 0, 1}),
                       cornell_offset({556.0, 548.8, 0, 1}), 
                       cornell_offset({0, 548.8, 559.2, 1}),
                       cornell_offset({556.0, 548.8, 559.2, 1}));
    ceiling->obj_mat = std::make_shared<material_diffuse>(white);
    obj_list.push_back(ceiling);

    trig_mesh *back = new trig_mesh(); 
    add_rect(*back, cornell_offset({0, 548.8, 559.2, 1}),
                    cornell_offset({556.0, 548.8, 559.2, 1}), 
                    cornell_offset({0, 0, 559.2, 1}),
                    cornell_offset({549.6, 0, 559.2, 1}));
    back->obj_mat = std::make_shared<material_diffuse>(white);
    obj_list.push_back(back);
   
    trig_mesh *right = new trig_mesh(); 
    add_rect(*right, cornell_offset({556.0, 548.8, 559.2, 1}),
                     cornell_offset({556.0, 548.8, 0, 1}), 
                     cornell_offset({549.6, 0, 559.2, 1}),
                     cornell_offset({552.8, 0, 0, 1}));
    right->obj_mat = std::make_shared<material_diffuse>(green);
    obj_list.push_back(right);

    trig_mesh *left = new trig_mesh(); 
    add_rect(*left,  cornell_offset({0, 548.8, 0, 1}),
                     cornell_offset({0, 548.8, 559.2, 1}), 
                     cornell_offset({0, 0, 0, 1}),
                     cornell_offset({0, 0, 559.2, 1}));
    left->obj_mat = std::make_shared<material_diffuse>(red);
    obj_list.push_back(left);

    trig_mesh *short_block = new trig_mesh(); 
    short_block->add_vert({258, 165, 114, 1});
    short_block->add_vert({414, 165, 65, 1});
    short_block->add_vert({258, 0, 114, 1});
    short_block->add_vert({414, 0, 65, 1});

    short_block->add_vert({308, 165, 272, 1});
    short_block->add_vert({464, 165, 225, 1});
    short_block->add_vert({308, 0, 272, 1});
    short_block->add_vert({464, 0, 225, 1});

    short_block->add_rect(0, 1, 2, 3);
    short_block->add_rect(5, 4, 7, 6);
    short_block->add_rect(4, 0, 6, 2);
    short_block->add_rect(1, 5, 3, 7);
    short_block->add_rect(4, 5, 0, 1);

    short_block->obj_mat = std::make_shared<material_diffuse>(white);

    cornell_offset(*short_block);

    obj_list.push_back(short_block);

    
    trig_mesh *tall_block = new trig_mesh(); 
    tall_block->add_vert({125, 330, 247, 1});
    tall_block->add_vert({282, 330, 296, 1});
    tall_block->add_vert({125, 0, 247, 1});
    tall_block->add_vert({282, 0, 296, 1});

    tall_block->add_vert({72, 330, 406, 1});
    tall_block->add_vert({233, 330, 456, 1});
    tall_block->add_vert({72, 0, 406, 1});
    tall_block->add_vert({233, 0, 456, 1});

    tall_block->add_rect(0, 1, 2, 3);
    tall_block->add_rect(5, 4, 7, 6);
    tall_block->add_rect(4, 0, 6, 2);
    tall_block->add_rect(1, 5, 3, 7);
    tall_block->add_rect(4, 5, 0, 1);

    tall_block->obj_mat = std::make_shared<material_diffuse>(white);

    cornell_offset(*tall_block);

    obj_list.push_back(tall_block); 
}