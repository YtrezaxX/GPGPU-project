#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

// Parameters inferred from slides or standard defaults
#define K_RESERVOIR 10
#define RGB_DIFF_THRESHOLD 110
#define TH_LOW 15
#define TH_HIGH 60
#define OPENING_RADIUS 3
#define WEIGHT_CAP 20.0f
#define WEIGHT_INC 3.0f

struct ReservoirSample {
    rgb8 color = {0, 0, 0};
    float weight = 0.0f;
};

struct PixelState {
    ReservoirSample samples[K_RESERVOIR];
    std::mt19937 rng;

    PixelState() {
        // Initialize with a default seed, will be re-seeded per pixel if needed
        // or just rely on global randomness if performance is an issue.
        // However, slides say "Maintain a random state for each pixel position".
        // We will seed it once based on position.
    }
};

// Global state variables
static std::vector<PixelState> g_state;
static int g_width = 0;
static int g_height = 0;

// Helper to calculate color difference
inline int color_diff(const rgb8& c1, const rgb8& c2) {
    return std::abs((int)c1.r - (int)c2.r) +
           std::abs((int)c1.g - (int)c2.g) +
           std::abs((int)c1.b - (int)c2.b);
}

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

void compute_cpp(ImageView<rgb8> in)
{
    // 1. Initialization / Resize
    if (g_width != in.width || g_height != in.height) {
        g_width = in.width;
        g_height = in.height;
        g_state.resize(g_width * g_height);
        
        // Initialize states
        for (int i = 0; i < g_width * g_height; ++i) {
             g_state[i].rng.seed(i); // Simple seeding
             for(int k=0; k<K_RESERVOIR; ++k) {
                 g_state[i].samples[k].weight = 0.0f;
                 g_state[i].samples[k].color = {0,0,0};
             }
        }
    }

    // Temporary buffers for processing
    // We need 'motion_score' (grayscale)
    std::vector<uint8_t> motion_score(g_width * g_height);
    
    // 2. Background Estimation Process (Pixel-wise)
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            int idx = y * in.width + x;
            PixelState& state = g_state[idx];
            rgb8 current_color = in.buffer[idx];

            // 2.1 Find matching reservoir
            int match_idx = -1;
            for (int k = 0; k < K_RESERVOIR; ++k) {
                if (state.samples[k].weight > 0 && color_diff(current_color, state.samples[k].color) < RGB_DIFF_THRESHOLD) {
                    match_idx = k;
                    break;
                }
            }

            // 2.2 Update weights and samples
            if (match_idx != -1) {
                // Match found: increase weight
                state.samples[match_idx].weight += WEIGHT_INC;
                if (state.samples[match_idx].weight > WEIGHT_CAP) {
                    state.samples[match_idx].weight = WEIGHT_CAP;
                }
            } else {
                // No match: find empty or replace low weight
                int empty_idx = -1;
                for (int k = 0; k < K_RESERVOIR; ++k) {
                    if (state.samples[k].weight <= 0.0f) {
                        empty_idx = k;
                        break;
                    }
                }

                if (empty_idx != -1) {
                    state.samples[empty_idx].color = current_color;
                    state.samples[empty_idx].weight = WEIGHT_INC; // Initial weight
                } else {
                    // Weighted reservoir replacement (simplified)
                    // Find sample with minimum weight to replace
                    int min_idx = 0;
                    float min_w = state.samples[0].weight;
                    for(int k=1; k<K_RESERVOIR; ++k) {
                        if (state.samples[k].weight < min_w) {
                            min_w = state.samples[k].weight;
                            min_idx = k;
                        }
                    }
                    
                    // Probabilistic replacement could be better, but deterministic min replacement is robust enough for simple cases
                    state.samples[min_idx].color = current_color;
                    state.samples[min_idx].weight = WEIGHT_INC; 
                }
            }

            // 2.3 Determine Background Color (Max Weight)
            int best_idx = -1;
            float max_w = -1.0f;
            for (int k = 0; k < K_RESERVOIR; ++k) {
                if (state.samples[k].weight > max_w) {
                    max_w = state.samples[k].weight;
                    best_idx = k;
                }
            }
            
            rgb8 bg_color = (best_idx != -1) ? state.samples[best_idx].color : rgb8{0,0,0};

            // 2.4 Calculate Motion Score (Difference)
            int diff = color_diff(current_color, bg_color);
            // Clamp to 255
            motion_score[idx] = (uint8_t)(diff > 255 ? 255 : diff);
        }
    }

    // 3. Mask Cleaning Process
    // Optimization: Linear time complexity O(W*H) independent of Radius.
    // Approach: Decompose 2D square opening into 1D Horizontal then 1D Vertical openings.
    // Algorithm: Van Herk / Gil-Werman (Dynamic Programming) for sliding window min/max.
    // Note: This approximates the "Disk" with a "Square" structuring element.
    
    // Helper for Transpose
    auto transpose = [](const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int w, int h) {
        dst.resize(w * h);
        // Blocked transpose could be faster, but simple is fine for now
        for(int y=0; y<h; ++y) {
            for(int x=0; x<w; ++x) {
                dst[x * h + y] = src[y * w + x];
            }
        }
    };

    // Helper for 1D sliding window (Van Herk / Gil-Werman)
    // Computes min (erosion) or max (dilation) for window size K = 2*R + 1
    // Input is assumed to be a flattened 2D image, processed row by row.
    auto sliding_window_1d = [&](const std::vector<uint8_t>& input, std::vector<uint8_t>& output, int w, int h, int r, bool is_dilation) {
        output.resize(w * h);
        int K = 2 * r + 1;
        
        // Buffers for prefix/suffix ops (reused per row)
        std::vector<uint8_t> L(w);
        std::vector<uint8_t> R_buf(w); // 'R' is reserved macro sometimes? use R_buf

        for (int y = 0; y < h; ++y) {
            const uint8_t* row_in = &input[y * w];
            uint8_t* row_out = &output[y * w];

            // 1. Fill L (prefix) and R_buf (suffix) based on blocks of size K
            for (int i = 0; i < w; i += K) {
                // Determine block boundaries
                int left = i;
                int right = std::min(i + K - 1, w - 1);
                
                // Prefix scan (L) from left to right
                uint8_t accum = row_in[left];
                L[left] = accum;
                for (int j = left + 1; j <= right; ++j) {
                    if (is_dilation) accum = std::max(accum, row_in[j]);
                    else             accum = std::min(accum, row_in[j]);
                    L[j] = accum;
                }

                // Suffix scan (R_buf) from right to left
                accum = row_in[right];
                R_buf[right] = accum;
                for (int j = right - 1; j >= left; --j) {
                    if (is_dilation) accum = std::max(accum, row_in[j]);
                    else             accum = std::min(accum, row_in[j]);
                    R_buf[j] = accum;
                }
            }

            // 2. Compute result for each pixel
            // The result at 'x' is the min/max of window [x-r, x+r].
            // This window has size K.
            // In VHGW, window [i, i+K-1] min is min(R_buf[i], L[i+K-1]).
            // Here, window starts at start_idx = x - r.
            for (int x = 0; x < w; ++x) {
                int start_idx = x - r;
                int end_idx = x + r;
                
                uint8_t val;
                
                if (start_idx < 0) {
                     // Window assumes padding? 
                     // Standard behavior: clamp to boundary or assume infinite extension.
                     // Let's Clamp indices to [0, w-1].
                     // But VHGW relies on fixed window size K logic.
                     // Fallback for boundaries or careful logic:
                     // Simplest: just iterate (naive) for boundaries, use fast for center.
                     // Or use the computed L/R with clamping?
                     // No, L/R property holds for full blocks.
                     
                     // Fallback to naive for borders to avoid segfaults/complexity
                     val = row_in[x]; // Init
                     int s = std::max(0, start_idx);
                     int e = std::min(w - 1, end_idx);
                     if (is_dilation) {
                         val = 0; 
                         for(int k=s; k<=e; ++k) val = std::max(val, row_in[k]);
                     } else {
                         val = 255;
                         for(int k=s; k<=e; ++k) val = std::min(val, row_in[k]);
                     }
                } else if (end_idx >= w) {
                     // Right boundary
                     val = row_in[x];
                     int s = std::max(0, start_idx);
                     int e = std::min(w - 1, end_idx);
                     if (is_dilation) {
                         val = 0;
                         for(int k=s; k<=e; ++k) val = std::max(val, row_in[k]);
                     } else {
                         val = 255;
                         for(int k=s; k<=e; ++k) val = std::min(val, row_in[k]);
                     }
                } else {
                     // Center: valid window of size K
                     // min( window [start_idx, start_idx + K - 1] )
                     // = op( R_buf[start_idx], L[start_idx + K - 1] )
                     // Note: start_idx + K - 1 should be end_idx
                     if (is_dilation) val = std::max(R_buf[start_idx], L[end_idx]);
                     else             val = std::min(R_buf[start_idx], L[end_idx]);
                }
                row_out[x] = val;
            }
        }
    };

    // Full separable morphology
    auto morphology_separable = [&](const std::vector<uint8_t>& input, std::vector<uint8_t>& output, bool is_dilation) {
        int r = OPENING_RADIUS;
        if (r == 0) {
            output = input;
            return;
        }

        std::vector<uint8_t> intermediate(g_width * g_height);
        std::vector<uint8_t> transposed_in(g_height * g_width); // W*H, just swapped dimensions logic
        std::vector<uint8_t> transposed_out(g_height * g_width);

        // 1. Horizontal Pass
        sliding_window_1d(input, intermediate, g_width, g_height, r, is_dilation);

        // 2. Vertical Pass (Transpose -> Horizontal -> Transpose)
        transpose(intermediate, transposed_in, g_width, g_height);
        
        // Process "Rows" of the transposed image (which are columns of original)
        // Width is now g_height, Height is now g_width
        sliding_window_1d(transposed_in, transposed_out, g_height, g_width, r, is_dilation);

        transpose(transposed_out, output, g_height, g_width);
    };

    // Buffer for intermediate result
    std::vector<uint8_t> temp_buf(g_width * g_height);

    // Erosion
    morphology_separable(motion_score, temp_buf, false);
    // Dilation
    morphology_separable(temp_buf, motion_score, true);

    // 3.2 Hysteresis thresholding
    // Propagate strong signals (>= TH_HIGH) to weak signals (>= TH_LOW)
    // Using a queue for flood fill
    std::vector<uint8_t> final_mask(g_width * g_height, 0);
    std::vector<int> q;
    q.reserve(g_width * g_height);

    // Find seeds
    for (int i = 0; i < g_width * g_height; ++i) {
        if (motion_score[i] >= TH_HIGH) {
            final_mask[i] = 255;
            q.push_back(i);
        }
    }

    // Propagate
    size_t head = 0;
    while(head < q.size()) {
        int idx = q[head++];
        int cx = idx % g_width;
        int cy = idx / g_width;

        // Check 4-connectivity or 8-connectivity? Usually 8.
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx >= 0 && nx < g_width && ny >= 0 && ny < g_height) {
                    int nidx = ny * g_width + nx;
                    if (final_mask[nidx] == 0 && motion_score[nidx] >= TH_LOW) {
                        final_mask[nidx] = 255;
                        q.push_back(nidx);
                    }
                }
            }
        }
    }

    // 4. Final Output (Visualization)
    // "input + 0.5 * red * mask"
    for (int i = 0; i < g_width * g_height; ++i) {
        if (final_mask[i] > 0) {
            // Apply red overlay
            // Original logic: in + 0.5 * red * mask.
            // Assuming mask is binary 255.
            // let's add 128 to red channel
            int new_r = (int)in.buffer[i].r + 128; 
            in.buffer[i].r = (uint8_t)(new_r > 255 ? 255 : new_r);
        }
    }
}


extern "C" {
  static double total_compute_time = 0;
  static int compute_count = 0;

  static Parameters g_params;

  void cpt_init(Parameters* params)
  {
    g_params = *params;
  }

  void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
  {
    auto img = ImageView<rgb8>{(rgb8*)buffer, width, height, stride};
    if (g_params.device == e_device_t::CPU)
    {
      auto start = std::chrono::high_resolution_clock::now();
      compute_cpp(img);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      total_compute_time += elapsed.count();
      compute_count++;
      if(compute_count % 10 == 0)
        std::cout << "AVERAGE_COMPUTE_TIME: " << (total_compute_time/compute_count) << " ms" << std::endl;
    }
    else if (g_params.device == e_device_t::GPU)
    {
      auto start = std::chrono::high_resolution_clock::now();
      compute_cu(img);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      total_compute_time += elapsed.count();
      compute_count++;
      if(compute_count % 10 == 0)
        std::cout << "AVERAGE_COMPUTE_TIME: " << (total_compute_time/compute_count) << " ms" << std::endl;
    }
  }

}
