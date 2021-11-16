#include <nbody/body.hpp>


BodyPool::BodyPool(size_t size, double position_range, double mass_range, int iteration_u):
        iteration{0},iteration_u{iteration_u},duration{0},bodies{} {

    std::random_device device;
    std::default_random_engine engine{device()};
    std::uniform_real_distribution<double> position_dist{0, position_range};
    std::uniform_real_distribution<double> mass_dist{0, mass_range};
    BodyPool::bodies.reserve(size);
    for(size_t i = 0; i<size; i++)
        BodyPool::bodies.push_back(Body(i,position_dist(engine),position_dist(engine),0.0,0.0,0.0,0.0,mass_dist(engine)));
    #ifdef MPI
    BodyPool::para = new Para;
    #endif
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

#ifdef MPI
void BodyPool::terminate_slave() {
    BodyPool::para->terminate = 1;
    MPI_Bcast(para, sizeof(Para), MPI_BYTE, 0, MPI_COMM_WORLD);
}
#endif
#ifdef MPI
void BodyPool::master_cal(double elapse,
                        double gravity,
                        double position_range,
                        double radius, size_t proc) {
    BodyPool::clear_acceleration();
    BodyPool::iteration++;
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<Body> snapshot = BodyPool::bodies;
    if(1==proc) {
        /** SEQUENTIAL PART **/
        for (auto &body: BodyPool::bodies) {
            cnu(body,radius,gravity,snapshot);
            body.update_for_tick(elapse, position_range, radius);
        }
    }
    else {
        /** MPI IMPLEMENTATION **/
        BodyPool::para->elapse = elapse;
        BodyPool::para->gravity = gravity;
        BodyPool::para->position_range = position_range;
        BodyPool::para->radius = radius;
        BodyPool::para->size = size();
        BodyPool::para->terminate = 0;
        int* recvcounts = new int[proc];
        int* displs = new int[proc];
        int count = 0;
        recvcounts[0] = getLength(size(),proc,0)*sizeof(Body);
        displs[0] = 0;
        for(size_t i=1;i<proc;i++){
            count += recvcounts[i-1]; //size of each row: sizeof(int)*size
            recvcounts[i] = getLength(size(),proc,i)*sizeof(Body);
            displs[i] = count;
        }
        MPI_Bcast(BodyPool::para, sizeof(Para), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(snapshot.data(), size()*sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
        /** Let the master work as well **/
        std::vector<Body> my_partition;
        my_partition.resize(getLength(size(),proc,0));
        #ifdef HYBRID
        omp_set_num_threads(omp_get_max_threads());
        // std::cout<<"omp_get_max_threads PER task: "<<omp_get_max_threads()<<std::endl;
        #pragma omp parallel for default(shared)
        #endif
        for(size_t i=0; i<getLength(size(),proc,i); i++) {
            my_partition[i] = bodies[i];
            cnu(my_partition[i],para->radius,para->gravity,snapshot);
            my_partition[i].update_for_tick(para->elapse, para->position_range, para->radius);
        }
        MPI_Gatherv(my_partition.data(),recvcounts[0],MPI_BYTE,bodies.data(),recvcounts,displs,MPI_BYTE,0,MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
}
#endif

#ifdef PTHREAD
void BodyPool::master_cal(double elapse,
                        double gravity,
                        double position_range,
                        double radius, size_t proc) {

    BodyPool::clear_acceleration();
    BodyPool::iteration++;
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<Body> snapshot = BodyPool::bodies;
    if(1==proc) {
        /** SEQUENTIAL PART **/
        for (auto &body: BodyPool::bodies) {
            cnu(body,radius,gravity,snapshot);
            body.update_for_tick(elapse, position_range, radius);
        }
    }
    else {
        Para* para = new Para[proc];
        pthread_t* threads = new pthread_t[proc];
        for(int i=0;i<static_cast<int>(proc);i++){
            para[i].elapse = elapse;
            para[i].gravity = gravity;
            para[i].position_range = position_range;
            para[i].radius = radius;
            para[i].proc = proc;
            para[i].tid = i;
            para[i].snapshot_ptr = &snapshot;
            para[i].bodies_ptr = &(BodyPool::bodies);
            pthread_create(threads+i,NULL,BodyPool::slave_cal,para+i);
        }
        for(int i=0;i<static_cast<int>(proc);i++) pthread_join(threads[i],NULL);
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
}
#endif


#ifdef OPENMP
void BodyPool::master_cal(double elapse,
                        double gravity,
                        double position_range,
                        double radius, size_t proc) {
    BodyPool::clear_acceleration();
    BodyPool::iteration++;
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<Body> snapshot = BodyPool::bodies;
    omp_set_num_threads(proc);
    #pragma omp parallel for shared(bodies,radius,gravity,snapshot,elapse, position_range) default(none)
    for(size_t i = 0; i<bodies.size(); ++i) {
        cnu(bodies[i],radius,gravity,snapshot);
        bodies[i].update_for_tick(elapse, position_range, radius);
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
}
#endif