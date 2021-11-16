#pragma once

#include <random>
#include <utility>
#include <iostream>
#ifdef MPI
#include <mpi.h>
#endif
#ifdef PTHREAD
#include <pthread.h>
#endif
#if defined(OPENMP) || defined(HYBRID)
#include <omp.h>
#endif



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




#ifdef MPI
struct Para {
    double radius;
    double gravity;
    double elapse;
    double position_range;
    char terminate;
    int size;
};
#endif

#ifdef PTHREAD
struct Para {
    double radius;
    double gravity;
    double elapse;
    double position_range;
    int tid;
    int proc;
    std::vector<Body>* bodies_ptr;
    std::vector<Body>* snapshot_ptr;
};
#endif


class BodyPool {
    int iteration;
    int iteration_u;
    long duration;
    #ifdef MPI
    Para* para;
    #endif
    std::vector<Body> bodies;
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    static constexpr double COLLISION_RATIO = 0.001;
    public:

    static inline size_t getLength(const size_t &size, const size_t &proc, const size_t &rank){
        if(size<(proc-1)) return (rank-1)<size;
        return (size-rank+1)/(proc-1) + ((size-rank+1)%(proc-1) > 0); // ceil funct
    }

    BodyPool(size_t size, double position_range, double mass_range, int iteration_u);

    Body& get_body(size_t index);

    int get_iteration();

    int get_iteration_u();

    long get_duration();

    void clear_acceleration();

    size_t size();

    static void cnu(Body& i, double radius, double gravity, std::vector<Body>& snapshot){
    
    for(auto &j: snapshot){
        int sign_s = 1;
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
#ifdef MPI
    static void* slave_cal(size_t proc, size_t rank){

        std::vector<Body> my_partition;
        std::vector<Body> snapshot;
        Para* para = new Para;
        // std::cout<<"I am slave_cal from rank# "<<rank<<std::endl;
        while(true){
            MPI_Bcast(para, sizeof(Para), MPI_BYTE, 0, MPI_COMM_WORLD);
            if(para->terminate) break;
            size_t all_size = para->size;
            size_t my_size = BodyPool::getLength(all_size,proc,rank);
            size_t my_start = 0;
            for(size_t r=1;r<rank;r++) my_start += BodyPool::getLength(all_size,proc,r);
            //std::cout<<"my_start of rank#"<<rank<<" : "<<my_start<<std::endl;
            my_partition.resize(my_size);
            snapshot.resize(all_size);
            MPI_Bcast(snapshot.data(), all_size*sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
            #ifdef HYBRID
            omp_set_num_threads(omp_get_max_threads());
            // std::cout<<"omp_get_max_threads PER task: "<<omp_get_max_threads()<<std::endl;
            #pragma omp parallel for default(shared)
            #endif
            for(size_t i=0; i<my_size; i++) {
                my_partition[i] = snapshot[i+my_start];
                cnu(my_partition[i],para->radius,para->gravity,snapshot);
                //cnu(i,my_start,para->radius,para->gravity,my_partition,snapshot);
                my_partition[i].update_for_tick(para->elapse, para->position_range, para->radius);
            }
            MPI_Gatherv(my_partition.data(), my_size*sizeof(Body), MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
            }
        return NULL;
    }
#endif

#ifdef PTHREAD
    static void* slave_cal(void* ptr){
        Para* para = static_cast<Para*>(ptr);
        size_t all_size = para->snapshot_ptr->size();
        size_t my_size = BodyPool::getLength(all_size,para->proc,para->tid);
        size_t my_start = 0;
        for(int r=1;r<para->tid;r++) my_start += BodyPool::getLength(all_size,para->proc,r);
        for(size_t i=my_start;i<my_start+my_size;++i){
            cnu((*(para->bodies_ptr))[i],para->radius,para->gravity,*(para->snapshot_ptr));
            (*(para->bodies_ptr))[i].update_for_tick(para->elapse, para->position_range, para->radius);
        }
        return NULL;
    }
#endif
    void terminate_slave();
    void master_cal(double elapse,double gravity,double position_range,double radius, size_t proc);


};



