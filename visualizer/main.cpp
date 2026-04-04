#include "image.h"
#include "scanline.h"
#include "dct.h"
#include "decoder.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------
struct AppState {
    // Image
    std::string image_path;
    bc::Image image;
    GLuint texture_id = 0;
    bool image_loaded = false;

    // Region (defaults to full image)
    float region_x = 0, region_y = 0, region_w = 0, region_h = 0;
    float dir_x = 1.0f, dir_y = 0.0f;

    // Pipeline params
    int filter_type_idx = 1; // 0=none, 1=lowpass, 2=hard, 3=soft, 4=bandpass
    float cutoff = 0.3f;
    float threshold = 10.0f;
    int band_low = 0;
    int band_high = 50;
    int num_scanlines = 5;
    float scanline_spacing = 2.0f;

    // Pipeline results
    std::vector<bc::Scanline> raw_scanlines;
    bc::Scanline averaged;
    std::vector<float> dct_coeffs;
    std::vector<float> spectrum;
    std::vector<float> filtered_coeffs;
    bc::Scanline filtered;
    std::vector<bc::Edge> edges;
    bc::DecodeResult result;

    // UI
    bool show_power_spectrum = false;
    bool pipeline_dirty = true;
    char path_buf[512] = {};
};

// ---------------------------------------------------------------------------
// GL texture helpers
// ---------------------------------------------------------------------------
static GLuint upload_texture(const bc::Image& img) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, img.width, img.height, 0,
                 GL_RED, GL_UNSIGNED_BYTE, img.pixels.data());
    return tex;
}

static void delete_texture(GLuint& tex) {
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
}

// ---------------------------------------------------------------------------
// Load image into state
// ---------------------------------------------------------------------------
static void load_image_into_state(AppState& s, const std::string& path) {
    try {
        s.image = bc::load_image(path);
        s.image_path = path;
        s.image_loaded = true;

        delete_texture(s.texture_id);
        s.texture_id = upload_texture(s.image);

        // Default region: full image, horizontal scan
        s.region_x = 0;
        s.region_y = 0;
        s.region_w = static_cast<float>(s.image.width);
        s.region_h = static_cast<float>(s.image.height);
        s.dir_x = 1.0f;
        s.dir_y = 0.0f;

        s.pipeline_dirty = true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load image: %s\n", e.what());
        s.image_loaded = false;
    }
}

// ---------------------------------------------------------------------------
// Run the decode pipeline
// ---------------------------------------------------------------------------
static void run_pipeline(AppState& s) {
    if (!s.image_loaded) return;

    bc::BarcodeRegion region{
        s.region_x, s.region_y, s.region_w, s.region_h,
        bc::Vec2f{s.dir_x, s.dir_y}.normalized()
    };

    bc::ExtractionParams ep;
    ep.num_scanlines = s.num_scanlines;
    ep.scanline_spacing = s.scanline_spacing;

    s.raw_scanlines = bc::extract_scanlines(s.image, region, ep);
    s.averaged = bc::average_scanlines(s.raw_scanlines);

    // DCT
    s.dct_coeffs = bc::dct_ii(s.averaged);
    s.spectrum = bc::power_spectrum(s.dct_coeffs);

    // Filter
    if (s.filter_type_idx == 0) {
        // No filter
        s.filtered_coeffs = s.dct_coeffs;
        s.filtered = s.averaged;
    } else {
        bc::FilterParams fp;
        switch (s.filter_type_idx) {
            case 1: fp.type = bc::FilterType::LowPass;       fp.cutoff = s.cutoff; break;
            case 2: fp.type = bc::FilterType::HardThreshold;  fp.threshold = s.threshold; break;
            case 3: fp.type = bc::FilterType::SoftThreshold;  fp.threshold = s.threshold; break;
            case 4:
                fp.type = bc::FilterType::BandPass;
                fp.band_low = s.band_low;
                fp.band_high = std::min(s.band_high, static_cast<int>(s.dct_coeffs.size()) - 1);
                break;
        }
        s.filtered_coeffs = bc::apply_filter(s.dct_coeffs, fp);
        s.filtered = bc::dct_iii(s.filtered_coeffs);
    }

    // Edge detection & decode
    s.edges = bc::detect_edges(s.filtered);
    s.result = bc::decode_scanline(s.filtered);

    s.pipeline_dirty = false;
}

// ---------------------------------------------------------------------------
// Draw a plot with custom overlay (edges, etc.)
// ---------------------------------------------------------------------------
static void plot_scanline(const char* label, const bc::Scanline& data,
                          float height = 100.0f,
                          const std::vector<bc::Edge>* edges_to_draw = nullptr) {
    if (data.empty()) return;

    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    float vmin = *min_it;
    float vmax = *max_it;
    if (vmax - vmin < 1.0f) { vmin -= 1.0f; vmax += 1.0f; }

    ImGui::PlotLines(label, data.data(), static_cast<int>(data.size()),
                     0, nullptr, vmin, vmax, ImVec2(-1, height));

    // Draw edge markers on the last PlotLines widget
    if (edges_to_draw && !edges_to_draw->empty()) {
        ImVec2 plot_min = ImGui::GetItemRectMin();
        ImVec2 plot_max = ImGui::GetItemRectMax();
        float plot_w = plot_max.x - plot_min.x;
        float n = static_cast<float>(data.size());

        auto* draw_list = ImGui::GetWindowDrawList();
        for (auto& e : *edges_to_draw) {
            float t = e.position / (n - 1.0f);
            float x = plot_min.x + t * plot_w;
            ImU32 color = e.rising ? IM_COL32(0, 200, 0, 180) : IM_COL32(200, 0, 0, 180);
            draw_list->AddLine(ImVec2(x, plot_min.y), ImVec2(x, plot_max.y), color, 1.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    // OpenGL 3.3 core
#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    const char* glsl_version = "#version 150";
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    const char* glsl_version = "#version 330";
#endif

    GLFWwindow* window = glfwCreateWindow(1400, 900, "Barcode Visualizer", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    AppState state;

    // Load image from CLI arg if provided
    if (argc > 1) {
        std::snprintf(state.path_buf, sizeof(state.path_buf), "%s", argv[1]);
        load_image_into_state(state, argv[1]);
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Run pipeline if dirty
        if (state.pipeline_dirty && state.image_loaded) {
            run_pipeline(state);
        }

        // Full-window ImGui window
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::Begin("##Main", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        // -- Top controls bar --
        ImGui::PushItemWidth(300);
        if (ImGui::InputText("Image Path", state.path_buf, sizeof(state.path_buf),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            load_image_into_state(state, state.path_buf);
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            load_image_into_state(state, state.path_buf);
        }

        ImGui::Separator();

        // Filter controls
        {
            const char* filter_names[] = {"None", "Low Pass", "Hard Threshold",
                                          "Soft Threshold", "Band Pass"};
            ImGui::PushItemWidth(130);
            if (ImGui::Combo("Filter", &state.filter_type_idx, filter_names, 5)) {
                state.pipeline_dirty = true;
            }
            ImGui::PopItemWidth();

            ImGui::SameLine();
            ImGui::PushItemWidth(150);
            if (state.filter_type_idx == 1) {
                if (ImGui::SliderFloat("Cutoff", &state.cutoff, 0.01f, 1.0f, "%.2f"))
                    state.pipeline_dirty = true;
            } else if (state.filter_type_idx == 2 || state.filter_type_idx == 3) {
                if (ImGui::SliderFloat("Threshold", &state.threshold, 0.0f, 100.0f, "%.1f"))
                    state.pipeline_dirty = true;
            } else if (state.filter_type_idx == 4) {
                if (ImGui::SliderInt("Band Low", &state.band_low, 0, 200))
                    state.pipeline_dirty = true;
                ImGui::SameLine();
                if (ImGui::SliderInt("Band High", &state.band_high, 0, 200))
                    state.pipeline_dirty = true;
            }
            ImGui::PopItemWidth();

            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::SliderInt("Scanlines", &state.num_scanlines, 1, 20))
                state.pipeline_dirty = true;
            ImGui::PopItemWidth();

            ImGui::SameLine();
            ImGui::PushItemWidth(100);
            if (ImGui::SliderFloat("Spacing", &state.scanline_spacing, 0.5f, 10.0f, "%.1f"))
                state.pipeline_dirty = true;
            ImGui::PopItemWidth();
        }

        ImGui::Separator();

        // -- Main content: left = image, right = plots --
        float avail_w = ImGui::GetContentRegionAvail().x;
        float avail_h = ImGui::GetContentRegionAvail().y - 30.0f; // reserve for result bar
        float left_w = avail_w * 0.35f;
        float right_w = avail_w - left_w - ImGui::GetStyle().ItemSpacing.x;

        // Left panel: image view
        ImGui::BeginChild("ImagePanel", ImVec2(left_w, avail_h), ImGuiChildFlags_Borders);
        if (state.image_loaded && state.texture_id) {
            ImGui::Text("Image: %dx%d", state.image.width, state.image.height);

            // Region controls
            bool region_changed = false;
            ImGui::PushItemWidth(left_w - 15.0f);
            region_changed |= ImGui::DragFloat("X", &state.region_x, 1.0f, 0.0f,
                                                static_cast<float>(state.image.width));
            region_changed |= ImGui::DragFloat("Y", &state.region_y, 1.0f, 0.0f,
                                                static_cast<float>(state.image.height));
            region_changed |= ImGui::DragFloat("W", &state.region_w, 1.0f, 1.0f,
                                                static_cast<float>(state.image.width));
            region_changed |= ImGui::DragFloat("H", &state.region_h, 1.0f, 1.0f,
                                                static_cast<float>(state.image.height));
            bool dir_changed = false;
            dir_changed |= ImGui::DragFloat("Dir X", &state.dir_x, 0.01f, -1.0f, 1.0f);
            dir_changed |= ImGui::DragFloat("Dir Y", &state.dir_y, 0.01f, -1.0f, 1.0f);
            ImGui::PopItemWidth();
            if (region_changed || dir_changed) state.pipeline_dirty = true;

            ImGui::Separator();

            // Display image with overlays
            float img_w = static_cast<float>(state.image.width);
            float img_h = static_cast<float>(state.image.height);
            float display_w = ImGui::GetContentRegionAvail().x;
            float scale = display_w / img_w;
            float display_h = img_h * scale;

            ImVec2 cursor = ImGui::GetCursorScreenPos();
            ImGui::Image(static_cast<ImTextureID>(state.texture_id),
                         ImVec2(display_w, display_h));

            // Draw region rect overlay
            auto* draw_list = ImGui::GetWindowDrawList();
            ImVec2 r_min(cursor.x + state.region_x * scale,
                         cursor.y + state.region_y * scale);
            ImVec2 r_max(cursor.x + (state.region_x + state.region_w) * scale,
                         cursor.y + (state.region_y + state.region_h) * scale);
            draw_list->AddRect(r_min, r_max, IM_COL32(255, 255, 0, 200), 0.0f, 0, 2.0f);

            // Draw scanline positions
            if (!state.raw_scanlines.empty()) {
                auto dir = bc::Vec2f{state.dir_x, state.dir_y}.normalized();
                auto perp = dir.perpendicular();
                float center_offset = (state.num_scanlines - 1) * state.scanline_spacing * 0.5f;

                for (int i = 0; i < state.num_scanlines; ++i) {
                    float offset = i * state.scanline_spacing - center_offset;
                    float cy = state.region_y + state.region_h * 0.5f + perp.y * offset;
                    float cx = state.region_x + perp.x * offset;

                    ImVec2 p0(cursor.x + cx * scale,
                              cursor.y + cy * scale);
                    ImVec2 p1(cursor.x + (cx + dir.x * state.region_w) * scale,
                              cursor.y + (cy + dir.y * state.region_w) * scale);

                    ImU32 col = IM_COL32(0, 200, 255, 120);
                    draw_list->AddLine(p0, p1, col, 1.0f);
                }
            }
        } else {
            ImGui::TextWrapped("No image loaded. Enter a path above and click Load.");
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right panel: plots
        ImGui::BeginChild("PlotsPanel", ImVec2(right_w, avail_h), ImGuiChildFlags_Borders);
        if (state.image_loaded && !state.averaged.empty()) {
            float plot_h = (avail_h - 120.0f) / 4.0f; // 4 plots

            // 1. Raw scanlines (overlaid)
            ImGui::Text("Raw Scanlines (%d)", static_cast<int>(state.raw_scanlines.size()));
            if (!state.raw_scanlines.empty()) {
                // Find global min/max across all raw scanlines
                float vmin = 255.0f, vmax = 0.0f;
                for (auto& sl : state.raw_scanlines) {
                    for (float v : sl) {
                        vmin = std::min(vmin, v);
                        vmax = std::max(vmax, v);
                    }
                }
                if (vmax - vmin < 1.0f) { vmin -= 1.0f; vmax += 1.0f; }

                // Draw first scanline to establish plot rect
                ImGui::PlotLines("##raw0", state.raw_scanlines[0].data(),
                                 static_cast<int>(state.raw_scanlines[0].size()),
                                 0, nullptr, vmin, vmax, ImVec2(-1, plot_h));

                // Overlay additional scanlines using ImDrawList
                if (state.raw_scanlines.size() > 1) {
                    ImVec2 pmin = ImGui::GetItemRectMin();
                    ImVec2 pmax = ImGui::GetItemRectMax();
                    float pw = pmax.x - pmin.x;
                    float ph = pmax.y - pmin.y;
                    auto* dl = ImGui::GetWindowDrawList();

                    for (size_t si = 1; si < state.raw_scanlines.size(); ++si) {
                        auto& sl = state.raw_scanlines[si];
                        int n = static_cast<int>(sl.size());
                        if (n < 2) continue;

                        ImU32 col = IM_COL32(100, 180, 255,
                                             static_cast<int>(80 + 40 * (si % 4)));
                        for (int j = 0; j < n - 1; ++j) {
                            float t0 = static_cast<float>(j) / (n - 1);
                            float t1 = static_cast<float>(j + 1) / (n - 1);
                            float y0 = 1.0f - (sl[j] - vmin) / (vmax - vmin);
                            float y1 = 1.0f - (sl[j + 1] - vmin) / (vmax - vmin);
                            dl->AddLine(
                                ImVec2(pmin.x + t0 * pw, pmin.y + y0 * ph),
                                ImVec2(pmin.x + t1 * pw, pmin.y + y1 * ph),
                                col, 1.0f);
                        }
                    }
                }
            }

            ImGui::Spacing();

            // 2. Averaged scanline
            ImGui::Text("Averaged Scanline (%d samples)", static_cast<int>(state.averaged.size()));
            plot_scanline("##averaged", state.averaged, plot_h);

            ImGui::Spacing();

            // 3. DCT Coefficients / Power Spectrum
            ImGui::Checkbox("Power Spectrum", &state.show_power_spectrum);
            if (state.show_power_spectrum) {
                ImGui::Text("Power Spectrum (%d coefficients)", static_cast<int>(state.spectrum.size()));
                if (!state.spectrum.empty()) {
                    auto [mn, mx] = std::minmax_element(state.spectrum.begin(), state.spectrum.end());
                    ImGui::PlotHistogram("##spectrum", state.spectrum.data(),
                                         static_cast<int>(state.spectrum.size()),
                                         0, nullptr, 0.0f, *mx, ImVec2(-1, plot_h));
                }
            } else {
                ImGui::Text("DCT Coefficients (%d)", static_cast<int>(state.dct_coeffs.size()));
                if (!state.dct_coeffs.empty()) {
                    auto [mn, mx] = std::minmax_element(state.dct_coeffs.begin(), state.dct_coeffs.end());
                    float absmax = std::max(std::abs(*mn), std::abs(*mx));
                    ImGui::PlotHistogram("##dct", state.dct_coeffs.data(),
                                         static_cast<int>(state.dct_coeffs.size()),
                                         0, nullptr, -absmax, absmax, ImVec2(-1, plot_h));
                }
            }

            ImGui::Spacing();

            // 4. Filtered scanline with edge markers
            ImGui::Text("Filtered Scanline (edges: %d)", static_cast<int>(state.edges.size()));
            plot_scanline("##filtered", state.filtered, plot_h, &state.edges);
        } else {
            ImGui::TextWrapped("Load an image to see the processing pipeline.");
        }
        ImGui::EndChild();

        // -- Result bar --
        ImGui::Separator();
        if (state.image_loaded) {
            if (state.result.success) {
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                    "Result: %s  %s  confidence: %.2f",
                    state.result.format.c_str(), state.result.text.c_str(),
                    state.result.confidence);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Decode failed");
            }
        }

        ImGui::End(); // ##Main

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    delete_texture(state.texture_id);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
