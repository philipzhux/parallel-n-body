#include <nbody/cuda.hpp>
#include <stdio.h>
__global__
void slave_cal(Body* bodies, Body* snapshot, int size, double elapse,double gravity,double position_range,double radius){
    int all_size = size;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index;i<all_size; i+=stride){
        cnu(bodies[i],radius,gravity,snapshot,size);
        bodies[i].update_for_tick(elapse, position_range, radius);
    }
}
__host__ __device__
Body::Body(): 
index{0}, x{0.0}, y{0.0}, vx{0.0}, vy{0.0}, ax{0.0}, ay{0.0}, m{0.0} {}

__host__ __device__
Body::Body(size_t index, double x, double y, double vx, double vy, double ax, double ay, double m):
        index(index), x(x), y(y), vx(vx), vy(vy), ax(ax), ay(ay), m(m) {}

__host__ __device__
size_t Body::get_index() {
    return Body::index;
}

__host__ __device__
double & Body::get_x() {
    return Body::x;
}

__host__ __device__
double & Body::get_y() {
    return Body::y;
}

__host__ __device__
double & Body::get_vx() {
    return Body::vx;
}

__host__ __device__
double & Body::get_vy() {
    return Body::vy;
}

__host__ __device__
double & Body::get_ax() {
    return Body::ax;
}

__host__ __device__
double & Body::get_ay() {
    return Body::ay;
}

__host__ __device__
double & Body::get_m() {
    return Body::m;
}

__host__ __device__
double Body::distance_square(Body &that) {
    auto delta_x = Body::get_x() - that.get_x();
    auto delta_y = Body::get_y() - that.get_y();
    return delta_x * delta_x + delta_y * delta_y;
}

__host__ __device__
double Body::distance(Body &that) {
    return std::sqrt(Body::distance_square(that));
}

__host__ __device__
double Body::delta_x(Body &that) {
    return Body::get_x() - that.get_x();
}

__host__ __device__
double Body::delta_y(Body &that) {
    return Body::get_y() - that.get_y();
}


// collision with wall
__host__ __device__
void Body::handle_wall_collision(double position_range, double radius) {
    bool flag = false;
    if (Body::get_x() <= radius) {
        flag = true;
        Body::get_x() = radius + radius * COLLISION_RATIO;
        Body::get_vx() = -Body::get_vx();
    } else if (get_x() >= position_range - radius) {
        flag = true;
        Body::get_x() = position_range - radius - radius * COLLISION_RATIO;
        Body::get_vx() = -Body::get_vx();
    }

    if (Body::get_y() <= radius) {
        flag = true;
        Body::get_y() = radius + radius * COLLISION_RATIO;
        Body::get_vy() = -Body::get_vy();
    } else if (Body::get_y() >= position_range - radius) {
        flag = true;
        Body::get_y() = position_range - radius - radius * COLLISION_RATIO;
        Body::get_vy() = -Body::get_vy();
    }
    if (flag) {
        Body::get_ax() = 0;
        Body::get_ay() = 0;
    }
}

__host__ __device__
void Body::update_for_tick(
        double elapse,
        double position_range,
        double radius) {
    Body::get_vx() += Body::get_ax() * elapse;
    Body::get_vy() += Body::get_ay() * elapse;
    Body::handle_wall_collision(position_range, radius);
    Body::get_x() += Body::get_vx() * elapse;
    Body::get_y() += Body::get_vy() * elapse;
    Body::handle_wall_collision(position_range, radius);
}

__device__ void cnu(Body& i, double radius, double gravity, Body* snapshot, int size)
{
    
    for(int h=0;h<size;++h){
        int sign_s = 1;
        Body& j = snapshot[h];
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

void master_cal_cu(double elapse,
    double gravity,
    double position_range,
    double radius, size_t proc, Body* bodies, int size, int block_dim, int g_dim) {
        Body* device_bodies;
        Body* device_snapshot;
        cudaMalloc((void**)&device_bodies,size*sizeof(Body));
        cudaMalloc((void**)&device_snapshot,size*sizeof(Body));
        cudaMemcpy(device_bodies, bodies, size*sizeof(Body), cudaMemcpyHostToDevice);
        cudaMemcpy(device_snapshot, bodies, size*sizeof(Body), cudaMemcpyHostToDevice);
        slave_cal<<<g_dim,block_dim>>>(device_bodies, device_snapshot,size,elapse,gravity,position_range,radius);
        cudaDeviceSynchronize();
        cudaMemcpy(bodies, device_bodies, size*sizeof(Body), cudaMemcpyDeviceToHost);
        cudaFree(device_bodies);
        cudaFree(device_snapshot);
}