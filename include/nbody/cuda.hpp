//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <utility>




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
        __host__ __device__ Body();
    
        __host__ __device__ Body(size_t index, double x, double y, double vx, double vy, double ax, double ay, double m);

        __host__ __device__ size_t get_index();

        __host__ __device__ double &get_x();

        __host__ __device__ double &get_y();

        __host__ __device__ double &get_vx();

        __host__ __device__ double &get_vy();

        __host__ __device__ double &get_ax();

        __host__ __device__ double &get_ay();

        __host__ __device__ double &get_m();

        __host__ __device__ double distance_square(Body &that);

        __host__ __device__ double distance(Body &that);

        __host__ __device__ double delta_x(Body &that);

        __host__ __device__ double delta_y(Body &that);

        __host__ __device__ bool collide(Body &that, double radius);

        // collision with wall
        __host__ __device__ void handle_wall_collision(double position_range, double radius);

        __host__ __device__ void update_for_tick(double elapse,double position_range,double radius);

    };


 __global__ void slave_cal(Body* bodies, Body* snapshot, int size, double elapse,double gravity,double position_range,double radius);

 __device__ void cnu(Body& i, double radius, double gravity, Body* snapshot, int size);