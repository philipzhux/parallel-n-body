#include <nbody/body.hpp>

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