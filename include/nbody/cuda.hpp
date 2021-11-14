//
// Created by schrodinger on 11/2/21.
//
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
        __host__ __device__
        Body();
    
        __host__ __device__
        Body(size_t index, double x, double y, double vx, double vy, double ax, double ay, double m);

        __host__ __device__
        size_t get_index();

        __host__ __device__
        double &get_x();

        __host__ __device__
        double &get_y();

        __host__ __device__
        double &get_vx();

        __host__ __device__
        double &get_vy();

        __host__ __device__
        double &get_ax();

        __host__ __device__
        double &get_ay();

        __host__ __device__
        double &get_m();

        __host__ __device__
        double distance_square(Body &that);

        __host__ __device__
        double distance(Body &that);

        __host__ __device__
        double delta_x(Body &that);

        __host__ __device__
        double delta_y(Body &that);

        __host__ __device__
        bool collide(Body &that, double radius);

        // collision with wall
        __host__ __device__
        void handle_wall_collision(double position_range, double radius);

        __host__ __device__
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
    __host__ __device__

    BodyPool(size_t size, double position_range, double mass_range, int iteration_u);
    __host__ __device__
    Body& get_body(size_t index);

    __host__ __device__
    int get_iteration();

    __host__ __device__
    int get_iteration_u();

    __host__ __device__
    long get_duration();

    __host__ __device__
    void clear_acceleration();

    __host__ __device__
    size_t size();

    __device__
    static void cnu(Body& i, double radius, double gravity, Body* snapshot){
    
    for(Body* j_ptr=snapshot;j_ptr;j_ptr++){
        int sign_s = 1;
        Body& j = *(j_ptr);
        if(j.get_index()==i.get_index()) continue;
        if(j.get_index()<i.get_index()) sign_s=-1;
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance = i.distance(j);
        auto distance_square = i.distance_square(j);
        //auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        if (distance < radius) {
            distance = radius;
        }
        auto scalar = gravity / distance_square / distance;
        i.get_ax()  -= sign_s * scalar * delta_x * j.get_m();
        i.get_ay() -= sign_s * scalar * delta_y * j.get_m();
    }        
    }


    __global__
    static void slave_cal(Body* bodies, Body* snapshot, int size, double elapse,double gravity,double position_range,double radius){
        int all_size = size;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int i=index;i<all_size; i+=stride){
            cnu(bodies[i],radius,gravity,snapshot);
            bodies[i].update_for_tick(elapse, position_range, radius);
        }
    }

    __host__
    void master_cal(double elapse,double gravity,double position_range,double radius, size_t proc);

};



