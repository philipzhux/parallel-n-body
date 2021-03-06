#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <stdlib.h>
#include <unistd.h>
template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}
int main(int argc, char **argv) {
    int rank, proc;
    #ifdef MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int res = MPI_Comm_size(MPI_COMM_WORLD, &proc);
    if (MPI_SUCCESS != res) {
        throw std::runtime_error("failed to get MPI world size");
    }
    #endif
    #ifdef PTHREAD
    rank = 0;
    proc = 16;
    #endif
    #ifdef OPENMP
    rank = 0;
    proc = 16;
    #endif
    if(0 != rank) {
        #ifdef MPI
        BodyPool::slave_cal(proc,rank);
        #endif
    }
    else {
        static int iteration_u = 500;
        static float gravity = 100;
        static float space = 800;
        static float radius = 2;
        static int bodies = 100;
        static float elapse = 0.001;
        static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
        static float max_mass = 50;
        static int t = 1;
        int c;
        int gui_flag = 0;
        opterr = 0;
        while ((c = getopt (argc, argv, "gi:b:e:t:")) != -1){
            switch (c)
            {
                case 'g':
                    gui_flag = 1;
                    break;
                case 'i':
                    iteration_u = atoi(optarg);
                    break;
                case 'b':
                    bodies = atoi(optarg);
                    break;
                case 'e':
                    elapse = atof(optarg);
                    break;
                case 't':
                    #if defined(OPENMP) || defined(PTHREAD)
                    proc = atoi(optarg);
                    #endif
                    break;
                case '?':
                    break;
                default:
                    break;
            }
        }
        static float current_space = space;
        static float current_max_mass = max_mass;
        static int current_bodies = bodies;
        static int current_iteration_u = iteration_u;
        std::cout<<"interation_u = "<<iteration_u<<std::endl;
        BodyPool pool(static_cast<size_t>(bodies), space, max_mass, iteration_u);
        if(gui_flag){
            graphic::GraphicContext context{
                #ifdef MPI
                "N-Body MPI"
                #endif
                #ifdef PTHREAD
                "N-Body Pthread"
                #endif
                #ifdef OPENMP
                "N-Body OpenMP"
                #endif
            };
            context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
                auto io = ImGui::GetIO();
                ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
                ImGui::SetNextWindowSize(io.DisplaySize);
                ImGui::Begin(
                #ifdef MPI
                "N-Body MPI",
                #endif
                #ifdef PTHREAD
                "N-Body Pthread",
                #endif
                #ifdef OPENMP
                "N-Body OpenMP",
                #endif
                nullptr,ImGuiWindowFlags_NoMove
                | ImGuiWindowFlags_NoCollapse
                | ImGuiWindowFlags_NoTitleBar
                | ImGuiWindowFlags_NoResize);
                ImDrawList *draw_list = ImGui::GetWindowDrawList();
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
                ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
                ImGui::DragFloat("Radius", &radius, 0.5, 0.5, 20, "%f");
                ImGui::DragInt("Bodies", &current_bodies, 1, 2, 1000, "%d");
                ImGui::DragInt("Iterations", &current_iteration_u, 100, 100, INT32_MAX, "%d");
                ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
                ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
                ImGui::ColorEdit4("Color", &color.x);
                /* update only upon change */
                if (current_space != space || current_bodies != bodies || current_max_mass != max_mass || current_iteration_u != iteration_u) {
                    space = current_space;
                    bodies = current_bodies;
                    max_mass = current_max_mass;
                    iteration_u = current_iteration_u;
                    pool = BodyPool{static_cast<size_t>(bodies), space, max_mass, iteration_u};
                }
                {
                    if(pool.get_iteration()==pool.get_iteration_u()) {
                        std::cout<<"BODIES : "<<current_bodies<<std::endl;
                        std::cout<<"ITERATION TIMES : "<<pool.get_iteration()<<std::endl;
                        std::cout<<"DURATION (ns) : "<<pool.get_duration()<<std::endl;
                        std::cout<<"DURATION PER ITER: "<<(pool.get_duration()/pool.get_iteration())<<std::endl;
                        #ifdef MPI
                        pool.terminate_slave();
                        #endif
                        t = 0;
                        context->quit();
                    }
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    //pool.update_for_tick(elapse, gravity, space, radius);
                    pool.master_cal(elapse, gravity, space, radius, proc);
                    //pool.update_for_tick(elapse, gravity, space, radius);
                    for (size_t i = 0; i < pool.size(); ++i) {
                        auto body = pool.get_body(i);
                        auto x = p.x + static_cast<float>(body.get_x());
                        auto y = p.y + static_cast<float>(body.get_y());
                        draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                    }
                }
                ImGui::End();
            });
        }
        else {
            while(true) {
                if(pool.get_iteration()==pool.get_iteration_u()) {
                    std::cout<<"BODIES : "<<current_bodies<<std::endl;
                    std::cout<<"ITERATION TIMES : "<<pool.get_iteration()<<std::endl;
                    std::cout<<"DURATION (ns) : "<<pool.get_duration()<<std::endl;
                    std::cout<<"DURATION PER ITER: "<<(pool.get_duration()/pool.get_iteration())<<std::endl;
                    #ifdef MPI
                    pool.terminate_slave();
                    #endif
                    t = 0;
                    break;
                }
                pool.master_cal(elapse, gravity, space, radius, proc);
            }
        }
        if(t) { 
            #ifdef MPI
            pool.terminate_slave();
            #endif
            std::cout<<"ITERATION TIMES : "<<pool.get_iteration()<<std::endl;
            std::cout<<"DURATION (ns) : "<<pool.get_duration()<<std::endl;
            std::cout<<"DURATION PER ITER: "<<(pool.get_duration()/pool.get_iteration())<<std::endl;
        }
    }
    #ifdef MPI
    MPI_Finalize();
    #endif
}




