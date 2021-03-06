#include <nbody/cuda_host.hpp>
#include <graphic/graphic.hpp>
#include <chrono>
//#include <cuda_runtime_api.h>


BodyPool::BodyPool(size_t size, double position_range, double mass_range, int iteration_u):
        iteration{0},iteration_u{iteration_u},duration{0},bodies{} {
    std::random_device device;
    std::default_random_engine engine{device()};
    std::uniform_real_distribution<double> position_dist{0, position_range};
    std::uniform_real_distribution<double> mass_dist{0, mass_range};
    BodyPool::bodies.reserve(size);
    for(size_t i = 0; i<size; i++)
        BodyPool::bodies.push_back(Body(i,position_dist(engine),position_dist(engine),0.0,0.0,0.0,0.0,mass_dist(engine)));
}

Body& BodyPool::get_body(size_t index) {
    return BodyPool::bodies.at(index);
}

int BodyPool::get_iteration() {
    return BodyPool::iteration;
}

int BodyPool::get_iteration_u() {
    return BodyPool::iteration_u;
}

long BodyPool::get_duration() {
    return BodyPool::duration;
}

void BodyPool::clear_acceleration() {
    for(auto &body: BodyPool::bodies) {
        body.get_ax() = 0.0;
        body.get_ay() = 0.0;
    }
}

size_t BodyPool::size() {
    return BodyPool::bodies.size();
}

void BodyPool::master_cal(double elapse,
                        double gravity,
                        double position_range,
                        double radius, size_t proc, int block_dim, int g_dim) {
    BodyPool::clear_acceleration();
    BodyPool::iteration++;
    auto begin = std::chrono::high_resolution_clock::now();
    master_cal_cu(elapse,gravity,position_range,radius,proc,&(bodies[0]),size(),block_dim, g_dim);
    auto end = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}