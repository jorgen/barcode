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
    int filter_type_idx = 1; // 0=none, 1=lowpass, 2=hard, 3=soft, 4=bandpass, 5=gaussian, 6=wiener, 7=wiener_deconv, 8=highboost
    float cutoff = 0.3f;
    float threshold = 10.0f;
    int band_low = 0;
    int band_high = 50;
    float sigma = 3.0f;
    float noise_power = 10.0f;
    float blur_sigma = 1.0f;
    float noise_ratio = 0.01f;
    float boost = 1.0f;
    float estimated_blur_sigma = 0.0f;
    int num_scanlines = 5;
    float scanline_spacing = 2.0f;
    int edge_method_idx = 0; // 0=threshold, 1=gradient
    float min_gradient = 0.0f;

    // Pipeline results
    std::vector<bc::Scanline> raw_scanlines;
    bc::Scanline averaged;
    std::vector<float> dct_coeffs;
    std::vector<float> spectrum;
    std::vector<float> filtered_coeffs;
    bc::Scanline filtered;
    std::vector<bc::Edge> edges;
    bc::DecodeResult result;

    // Full-image filtered view
    bc::Image filtered_image;
    GLuint filtered_texture_id = 0;
    bool filter_2d = false;

    // UI
    bool show_power_spectrum = false;
    bool pipeline_dirty = true;
    char path_buf[512] = {};

    // Region drag state
    bool dragging_region = false;
    bool drag_is_move = false;
    float drag_start_img_x = 0, drag_start_img_y = 0;
    float drag_orig_region_x = 0, drag_orig_region_y = 0;
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
// Build FilterParams from current UI state
// ---------------------------------------------------------------------------
static bc::FilterParams build_filter_params(const AppState& s) {
    bc::FilterParams fp;
    switch (s.filter_type_idx) {
        case 1: fp.type = bc::FilterType::LowPass;       fp.cutoff = s.cutoff; break;
        case 2: fp.type = bc::FilterType::HardThreshold;  fp.threshold = s.threshold; break;
        case 3: fp.type = bc::FilterType::SoftThreshold;  fp.threshold = s.threshold; break;
        case 4:
            fp.type = bc::FilterType::BandPass;
            fp.band_low = s.band_low;
            fp.band_high = s.band_high;
            break;
        case 5: fp.type = bc::FilterType::Gaussian; fp.sigma = s.sigma; break;
        case 6: fp.type = bc::FilterType::Wiener;   fp.noise_power = s.noise_power; break;
        case 7:
            fp.type = bc::FilterType::WienerDeconv;
            fp.blur_sigma = s.blur_sigma;
            fp.noise_ratio = s.noise_ratio;
            break;
        case 8:
            fp.type = bc::FilterType::HighBoost;
            fp.boost = s.boost;
            fp.sigma = s.sigma;
            break;
        default: break; // filter_type_idx == 0 handled by caller
    }
    return fp;
}

// ---------------------------------------------------------------------------
// Filter entire image row-by-row (and optionally column-by-column)
// ---------------------------------------------------------------------------
static void filter_full_image(AppState& s) {
    if (!s.image_loaded || s.filter_type_idx == 0) {
        delete_texture(s.filtered_texture_id);
        s.filtered_image = {};
        return;
    }

    auto fp = build_filter_params(s);
    int w = s.image.width;
    int h = s.image.height;

    s.filtered_image.width = w;
    s.filtered_image.height = h;
    s.filtered_image.pixels.resize(w * h);

    // Row-by-row filtering
    for (int y = 0; y < h; ++y) {
        bc::Scanline row(w);
        for (int x = 0; x < w; ++x)
            row[x] = static_cast<float>(s.image.pixels[y * w + x]);

        auto filtered_row = bc::dct_filter(row, fp);

        for (int x = 0; x < w; ++x)
            s.filtered_image.pixels[y * w + x] =
                static_cast<uint8_t>(std::clamp(filtered_row[x], 0.0f, 255.0f));
    }

    // Optional column-by-column filtering (separable 2D DCT)
    if (s.filter_2d) {
        for (int x = 0; x < w; ++x) {
            bc::Scanline col(h);
            for (int y = 0; y < h; ++y)
                col[y] = static_cast<float>(s.filtered_image.pixels[y * w + x]);

            auto filtered_col = bc::dct_filter(col, fp);

            for (int y = 0; y < h; ++y)
                s.filtered_image.pixels[y * w + x] =
                    static_cast<uint8_t>(std::clamp(filtered_col[y], 0.0f, 255.0f));
        }
    }

    // Upload to GL texture
    delete_texture(s.filtered_texture_id);
    s.filtered_texture_id = upload_texture(s.filtered_image);
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
        delete_texture(s.filtered_texture_id);
        s.filtered_image = {};
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
        auto fp = build_filter_params(s);
        if (fp.type == bc::FilterType::BandPass)
            fp.band_high = std::min(fp.band_high, static_cast<int>(s.dct_coeffs.size()) - 1);
        s.filtered_coeffs = bc::apply_filter(s.dct_coeffs, fp);
        s.filtered = bc::dct_iii(s.filtered_coeffs);
    }

    // Edge detection & decode
    auto edge_method = s.edge_method_idx == 1 ? bc::EdgeMethod::Gradient : bc::EdgeMethod::Threshold;
    if (edge_method == bc::EdgeMethod::Gradient) {
        s.edges = bc::detect_edges_gradient(s.filtered, s.min_gradient);
    } else {
        s.edges = bc::detect_edges(s.filtered);
    }
    s.result = bc::decode_scanline(s.filtered, edge_method);

    // Full-image filtered view
    filter_full_image(s);

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
                                          "Soft Threshold", "Band Pass", "Gaussian", "Wiener",
                                          "Wiener Deconv", "High Boost"};
            ImGui::PushItemWidth(130);
            if (ImGui::Combo("Filter", &state.filter_type_idx, filter_names, 9)) {
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
            } else if (state.filter_type_idx == 5) {
                if (ImGui::SliderFloat("Sigma", &state.sigma, 1.0f, 10.0f, "%.1f"))
                    state.pipeline_dirty = true;
            } else if (state.filter_type_idx == 6) {
                if (ImGui::SliderFloat("Noise Power", &state.noise_power, 0.0f, 1000.0f, "%.1f"))
                    state.pipeline_dirty = true;
            } else if (state.filter_type_idx == 7) {
                if (ImGui::SliderFloat("Blur Sigma", &state.blur_sigma, 0.1f, 10.0f, "%.2f"))
                    state.pipeline_dirty = true;
                ImGui::SameLine();
                float log_nr = std::log10(state.noise_ratio);
                if (ImGui::SliderFloat("Noise Ratio", &log_nr, -3.0f, 0.0f, "%.2f")) {
                    state.noise_ratio = std::pow(10.0f, log_nr);
                    state.pipeline_dirty = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Auto")) {
                    if (!state.dct_coeffs.empty()) {
                        state.estimated_blur_sigma = bc::estimate_blur_sigma(state.dct_coeffs);
                        if (state.estimated_blur_sigma > 0.0f) {
                            state.blur_sigma = state.estimated_blur_sigma;
                            state.pipeline_dirty = true;
                        }
                    }
                }
                ImGui::SameLine();
                ImGui::Text("Est: %.2f", state.estimated_blur_sigma);
            } else if (state.filter_type_idx == 8) {
                if (ImGui::SliderFloat("Boost", &state.boost, 0.0f, 5.0f, "%.2f"))
                    state.pipeline_dirty = true;
                ImGui::SameLine();
                if (ImGui::SliderFloat("Sigma", &state.sigma, 1.0f, 10.0f, "%.1f"))
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

            ImGui::SameLine();
            if (ImGui::Checkbox("2D Filter", &state.filter_2d))
                state.pipeline_dirty = true;
        }

        // Edge detection controls
        {
            const char* edge_names[] = {"Threshold", "Gradient"};
            ImGui::PushItemWidth(130);
            if (ImGui::Combo("Edge Method", &state.edge_method_idx, edge_names, 2))
                state.pipeline_dirty = true;
            ImGui::PopItemWidth();

            if (state.edge_method_idx == 1) {
                ImGui::SameLine();
                ImGui::PushItemWidth(150);
                if (ImGui::SliderFloat("Min Gradient", &state.min_gradient, 0.0f, 50.0f, "%.1f"))
                    state.pipeline_dirty = true;
                ImGui::PopItemWidth();
            }
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
            auto& style = ImGui::GetStyle();
            ImGui::PushItemWidth(-ImGui::CalcTextSize("Dir X").x - style.ItemInnerSpacing.x);
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

            // Display images with overlays
            float img_w = static_cast<float>(state.image.width);
            float img_h = static_cast<float>(state.image.height);
            float content_w = ImGui::GetContentRegionAvail().x;
            bool has_filtered = state.filtered_texture_id != 0;

            // Scale to fill panel width; cap height to avoid overflow
            float scale = content_w / img_w;
            float avail_img_h = ImGui::GetContentRegionAvail().y;
            float max_img_h = has_filtered ? (avail_img_h - 60.0f) * 0.5f : avail_img_h;
            float display_w = content_w;
            float display_h = std::min(img_h * scale, std::max(max_img_h, 1.0f));

            // Helper lambda to draw region rect + scanline overlays on the last image
            auto draw_overlays = [&](ImVec2 cursor) {
                auto* draw_list = ImGui::GetWindowDrawList();
                ImVec2 r_min(cursor.x + state.region_x * scale,
                             cursor.y + state.region_y * scale);
                ImVec2 r_max(cursor.x + (state.region_x + state.region_w) * scale,
                             cursor.y + (state.region_y + state.region_h) * scale);
                draw_list->AddRect(r_min, r_max, IM_COL32(255, 255, 0, 200), 0.0f, 0, 2.0f);

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
            };

            // Handle click-drag on the original image to move/create region
            auto handle_image_drag = [&](ImVec2 img_screen_pos) {
                bool hovered = ImGui::IsItemHovered();

                // Hit-test: "near edge" means inside region but within margin of a border
                auto near_region_edge = [&](float mx, float my) {
                    constexpr float margin = 6.0f; // pixels in image coords
                    bool inside = mx >= state.region_x && mx <= state.region_x + state.region_w &&
                                  my >= state.region_y && my <= state.region_y + state.region_h;
                    if (!inside) return false;
                    float dl = mx - state.region_x;
                    float dr = (state.region_x + state.region_w) - mx;
                    float dt = my - state.region_y;
                    float db = (state.region_y + state.region_h) - my;
                    return dl < margin || dr < margin || dt < margin || db < margin;
                };

                // Cursor feedback
                if (state.dragging_region && state.drag_is_move) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
                } else if (hovered) {
                    ImVec2 mouse = ImGui::GetIO().MousePos;
                    float mx = (mouse.x - img_screen_pos.x) / scale;
                    float my = (mouse.y - img_screen_pos.y) / scale;
                    if (near_region_edge(mx, my))
                        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
                }

                // Start drag
                if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    ImVec2 mouse = ImGui::GetIO().MousePos;
                    float mx = (mouse.x - img_screen_pos.x) / scale;
                    float my = (mouse.y - img_screen_pos.y) / scale;
                    state.dragging_region = true;
                    state.drag_start_img_x = mx;
                    state.drag_start_img_y = my;
                    if (near_region_edge(mx, my)) {
                        state.drag_is_move = true;
                        state.drag_orig_region_x = state.region_x;
                        state.drag_orig_region_y = state.region_y;
                    } else {
                        state.drag_is_move = false;
                    }
                }

                // Continue drag (works even if mouse leaves the image)
                if (state.dragging_region) {
                    ImVec2 mouse = ImGui::GetIO().MousePos;
                    float mx = (mouse.x - img_screen_pos.x) / scale;
                    float my = (mouse.y - img_screen_pos.y) / scale;

                    if (state.drag_is_move) {
                        float dx = mx - state.drag_start_img_x;
                        float dy = my - state.drag_start_img_y;
                        state.region_x = std::clamp(state.drag_orig_region_x + dx,
                                                     0.0f, std::max(img_w - state.region_w, 0.0f));
                        state.region_y = std::clamp(state.drag_orig_region_y + dy,
                                                     0.0f, std::max(img_h - state.region_h, 0.0f));
                    } else {
                        float x0 = std::clamp(std::min(state.drag_start_img_x, mx), 0.0f, img_w);
                        float y0 = std::clamp(std::min(state.drag_start_img_y, my), 0.0f, img_h);
                        float x1 = std::clamp(std::max(state.drag_start_img_x, mx), 0.0f, img_w);
                        float y1 = std::clamp(std::max(state.drag_start_img_y, my), 0.0f, img_h);
                        state.region_x = x0;
                        state.region_y = y0;
                        state.region_w = std::max(x1 - x0, 1.0f);
                        state.region_h = std::max(y1 - y0, 1.0f);
                    }
                    state.pipeline_dirty = true;

                    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
                        state.dragging_region = false;
                }
            };

            if (has_filtered) {
                ImGui::Text("Original");
                ImVec2 cursor = ImGui::GetCursorScreenPos();
                ImGui::Image(static_cast<ImTextureID>(state.texture_id),
                             ImVec2(display_w, display_h));
                // Overlay invisible button for robust click-drag input
                ImGui::SetCursorScreenPos(cursor);
                ImGui::InvisibleButton("##orig_interact", ImVec2(display_w, display_h));
                handle_image_drag(cursor);
                draw_overlays(cursor);

                ImGui::Spacing();

                ImGui::Text("Filtered%s", state.filter_2d ? " (2D)" : "");
                cursor = ImGui::GetCursorScreenPos();
                ImGui::Image(static_cast<ImTextureID>(state.filtered_texture_id),
                             ImVec2(display_w, display_h));
                draw_overlays(cursor);
            } else {
                ImVec2 cursor = ImGui::GetCursorScreenPos();
                ImGui::Image(static_cast<ImTextureID>(state.texture_id),
                             ImVec2(display_w, display_h));
                // Overlay invisible button for robust click-drag input
                ImGui::SetCursorScreenPos(cursor);
                ImGui::InvisibleButton("##orig_interact", ImVec2(display_w, display_h));
                handle_image_drag(cursor);
                draw_overlays(cursor);
            }
        } else {
            ImGui::TextWrapped("No image loaded. Enter a path above and click Load.");
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right panel: plots
        ImGui::BeginChild("PlotsPanel", ImVec2(right_w, avail_h), ImGuiChildFlags_Borders);
        if (state.image_loaded && !state.averaged.empty()) {
            // Non-plot overhead: 4 text labels (~80px) + 3 spacings (~24px) + 1 checkbox (~20px)
            float plots_avail = ImGui::GetContentRegionAvail().y;
            float plot_h = (plots_avail - 124.0f) / 4.0f;

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
                    "Result: %s  %s  confidence: %.2f  edges: %d",
                    state.result.format.c_str(), state.result.text.c_str(),
                    state.result.confidence, static_cast<int>(state.edges.size()));
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                    "Decode failed  edges: %d", static_cast<int>(state.edges.size()));
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
    delete_texture(state.filtered_texture_id);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
