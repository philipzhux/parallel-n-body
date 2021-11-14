#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/cuda.hpp>
template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}
int main(int argc, char **argv) {
    int proc;
    static int iteration_u = INT32_MAX;
    static float gravity = 100;
    static float space = 800;
    static float radius = 2;
    static int bodies = 100;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    static int current_iteration_u = iteration_u;
    static int t = 1;
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass, iteration_u);
    graphic::GraphicContext context{"Assignment 2"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
                    ImGuiWindowFlags_NoMove
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
                std::cout<<"ITERATION TIMES : "<<pool.get_iteration()<<std::endl;
                std::cout<<"DURATION (ns) : "<<pool.get_duration()<<std::endl;
                std::cout<<"DURATION PER ITER: "<<(pool.get_duration()/pool.get_iteration())<<std::endl;
                context->quit();
                #ifdef MPI
                pool.terminate_slave();
                #endif
                t = 0;
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
    if(t) { 
        std::cout<<"ITERATION TIMES : "<<pool.get_iteration()<<std::endl;
        std::cout<<"DURATION (ns) : "<<pool.get_duration()<<std::endl;
        std::cout<<"DURATION PER ITER: "<<(pool.get_duration()/pool.get_iteration())<<std::endl;
    }

}




