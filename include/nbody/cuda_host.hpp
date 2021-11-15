#pragma once

#include <random>
#include <utility>
#include <iostream>

class Body {
    size_t index;
    double x;
    double y;
    double vx;
    double vy;
    double ax;
    double ay;
    double m;
    static constexpr double COLLISION_RATIO = 0.01;

    public:
        Body();
    
        Body(size_t index, double x, double y, double vx, double vy, double ax, double ay, double m);

        size_t get_index();

        double &get_x();

        double &get_y();

        double &get_vx();

        double &get_vy();

        double &get_ax();

        double &get_ay();

        double &get_m();

        double distance_square(Body &that);

        double distance(Body &that);

        double delta_x(Body &that);

        double delta_y(Body &that);

        bool collide(Body &that, double radius);

        // collision with wall
        void handle_wall_collision(double position_range, double radius);

        void update_for_tick(double elapse,double position_range,double radius);

    };







class BodyPool {
    int iteration;
    int iteration_u;
    long duration;
    std::vector<Body> bodies;
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    static constexpr double COLLISION_RATIO = 0.001;
    public:
    BodyPool(size_t size, double position_range, double mass_range, int iteration_u);
    Body& get_body(size_t index);
    int get_iteration();
    int get_iteration_u();

    long get_duration();

    void clear_acceleration();
    size_t size();
    void master_cal(double elapse,
                        double gravity,
                        double position_range,
                        double radius, size_t proc, int block_dim, int g_dim);
};

void slave_cal(Body* bodies, Body* snapshot, int size, double elapse,double gravity,double position_range,double radius);


void master_cal_cu(double elapse,double gravity,double position_range,double radius, size_t proc, Body* bodies, int size, int block_dim, int g_dim);
