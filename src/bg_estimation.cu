#include "Compute.hpp"
#include "bg_estimation.hpp"
#include <algorithm>
#include <vector>
#include <iterator>
#include <curand_kernel.h>

// Buffer for the reservoirs
static ReservoirSampleBuffer* rsb = nullptr;

__global__ void random_state_init(ImageView<curandState> rng_states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = rng_states.width;
    int height = rng_states.height;

    int idx = y * width + x;
    if (idx < width * height)
    {
        curandState* row = &rng_states.buffer[idx];
        curand_init(42, idx, 0, row);
    } 
}


__global__ void reservoirs_init(ReservoirSampleBuffer* rsb, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (idx < width * height)
    {
        for (int k = 0; k < K_RESERVOIR; k++) {
            rsb[idx].rs[k].weight = 0.f;
            rsb[idx].rs[k].color = {0,0,0};
        }
    } 
}

// Helper to calculate color difference
__device__ inline int color_diff(const rgb8& c1, const rgb8& c2) {
    return std::abs((int)c1.r - (int)c2.r) +
           std::abs((int)c1.g - (int)c2.g) +
           std::abs((int)c1.b - (int)c2.b);
}

__device__ rgb8 update_color(rgb8 color1, rgb8 color2,  float weight) {
    rgb8 new_color;
    new_color.r = (float)color1.r * (weight - 1.f) / weight + (float)color2.r / weight;
    new_color.g = (float)color1.g * (weight - 1.f) / weight + (float)color2.g / weight;
    new_color.b = (float)color1.b * (weight - 1.f) / weight + (float)color2.b / weight;
    return new_color;
}

__device__ int find_matching_reservoir(rgb8 color, ReservoirSampleBuffer* rsb)
{
    // Default case: no empty reservoir + no matches
    int m_idx = -1;
    for (int i = 0; i < K_RESERVOIR; i++) {


        // Possibly best match
        if (rsb->rs[i].weight > 0)
        {
            // Match is found
            if (color_diff(color, rsb->rs[i].color) < RGB_DIFF_THRESHOLD)
            {
                return i;
            }
        }
        // Empty reservoir, match it
        else
        {
            m_idx = i;
        }
    }
    return m_idx;
}

__global__ void bg_estimation_kernel_opti0(ImageView<rgb8> in, ReservoirSampleBuffer* reservoirs)
{
    for (int y = 0; y < in.height; y++) {
        for (int x = 0; x < in.width; x++) {

            // Extract pixels from picture
            rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);

            // Init reservoir, extract color and find matching reservoir
            int idx = y * in.width + x;
            ReservoirSampleBuffer* rsb = &reservoirs[idx];
            rgb8 color = pixel_in[x];
            int m_idx = find_matching_reservoir(color, rsb);

            // Matched reservoir
            if (m_idx != -1 && rsb->rs[m_idx].weight != 0) {
                rsb->rs[m_idx].weight += 1;
                if (rsb->rs[m_idx].weight > MAX_WEIGHTS)
                    rsb->rs[m_idx].weight = MAX_WEIGHTS;
                rsb->rs[m_idx].color = update_color(rsb->rs[m_idx].color, color, rsb->rs[m_idx].weight);
            }

            // Empty slot
            else if (m_idx != -1 && rsb->rs[m_idx].weight == 0) {
                rsb->rs[m_idx].color = color;
                rsb->rs[m_idx].weight = 1;
            }

            // Weighted replacement
            else {
                float min_weight = rsb->rs[0].weight;
                int min_idx = 0;
                float total = 0;
                for (int k = 1; k < K_RESERVOIR; k++) {
                    if (rsb->rs[k].weight < min_weight)
                    {
                        min_weight = rsb->rs[k].weight;
                        min_idx = k;
                    }
                    total += rsb->rs[k].weight;
                }

                // Randomly replace
                float randomfloat = 0.2;
                if (randomfloat * total >= rsb->rs[min_idx].weight) {
                    rsb->rs[min_idx].weight = 1;
                    rsb->rs[min_idx].color = color;
                }
            }

            // Find the best color and save it
            float max_weight = -1.0;
            rgb8 best_color = {0,0,0};
            for (int k = 0; k < K_RESERVOIR; k++) {
                if (rsb->rs[k].weight > max_weight)
                {
                    max_weight = rsb->rs[k].weight;
                    best_color = rsb->rs[k].color;
                }
            }
            // Motion Score (Difference)
            int diff = color_diff(color, best_color);
            uint8_t green = (uint8_t)(diff > 255 ? 255 : diff);
            pixel_in[x] = {color.r, green, color.b};
            //pixel_in[x] = best_color;;
        }
    }
}

__global__ void bg_estimation_kernel_opti1(ImageView<rgb8> in, ReservoirSampleBuffer* reservoirs, curandState* rng_states)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < in.width && y < in.height) {

            // Extract pixels from picture
            rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);

            // Init reservoir, extract color and find matching reservoir
            int idx = y * in.width + x;
            ReservoirSampleBuffer* rsb = &reservoirs[idx];
            rgb8 color = pixel_in[x];
            int m_idx = find_matching_reservoir(color, rsb);

            // Matched reservoir
            if (m_idx != -1 && rsb->rs[m_idx].weight != 0) {
                rsb->rs[m_idx].weight += 1;
                if (rsb->rs[m_idx].weight > MAX_WEIGHTS)
                    rsb->rs[m_idx].weight = MAX_WEIGHTS;
                rsb->rs[m_idx].color = update_color(rsb->rs[m_idx].color, color, rsb->rs[m_idx].weight);
            }

            // Empty slot
            else if (m_idx != -1 && rsb->rs[m_idx].weight == 0) {
                rsb->rs[m_idx].color = color;
                rsb->rs[m_idx].weight = 1;
            }

            // Weighted replacement
            else {
                float min_weight = rsb->rs[0].weight;
                int min_idx = 0;
                float total = 0;
                for (int k = 1; k < K_RESERVOIR; k++) {
                    if (rsb->rs[k].weight < min_weight)
                    {
                        min_weight = rsb->rs[k].weight;
                        min_idx = k;
                    }
                    total += rsb->rs[k].weight;
                }

                // Randomly replace
                float randomfloat = curand_uniform(&rng_states[idx]);
                if (randomfloat * total >= rsb->rs[min_idx].weight) {
                    rsb->rs[min_idx].weight = 1;
                    rsb->rs[min_idx].color = color;
                }
            }

            // Find the best color and save it
            float max_weight = -1.0;
            rgb8 best_color = {0,0,0};
            for (int k = 0; k < K_RESERVOIR; k++) {
                if (rsb->rs[k].weight > max_weight)
                {
                    max_weight = rsb->rs[k].weight;
                    best_color = rsb->rs[k].color;
                }
            }
            
            // Motion Score (Difference)
            int diff = color_diff(color, best_color);
            uint8_t green = (uint8_t)(diff > 255 ? 255 : diff);
            pixel_in[x] = {color.r, green, color.b};
            //pixel_in[x] = best_color;
        }
}


// OPTI 2 ----------------------------------------------------------------

__device__ ReservoirSampleInfo find_matching_reservoir_struct(rgb8 color, ReservoirSampleBuffer* rsb)
{
    // Default case: no empty reservoir + no matches
    ReservoirSampleInfo info;
    float min_weight = rsb->rs[0].weight;
    bool found = false;
    info.min_idx = 0;
    for (int i = 0; i < K_RESERVOIR; i++) {

        float weight = rsb->rs[i].weight;

        // Update struct
        info.total += weight; // Total
        if (weight > info.best_weight) { // Best color
            info.best_weight = weight;
            info.best_color = rsb->rs[i].color;
        }
        if (weight < min_weight) { // Min weight
            min_weight = weight;
            info.min_idx = i;
        }


        // Possibly best match
        if (weight > 0)
        {
            // Match is found
            if (color_diff(color, rsb->rs[i].color) < RGB_DIFF_THRESHOLD && !found)
            {
                info.m_idx = i;
                found = true;
            }
        }
        // Empty reservoir, match it
        else if (!found)
        {
            info.m_idx = i;
        }
    }
    return info;
}

__global__ void bg_estimation_kernel_opti2(ImageView<rgb8> in, ReservoirSampleBuffer* reservoirs, curandState* rng_states)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < in.width && y < in.height) {

            // Extract pixels from picture
            rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);

            // Init reservoir, extract color and find matching reservoir
            int idx = y * in.width + x;
            ReservoirSampleBuffer* rsb = &reservoirs[idx];
            rgb8 color = pixel_in[x];

            ReservoirSampleInfo info = find_matching_reservoir_struct(color, rsb);
            int m_idx = info.m_idx;

            // Matched reservoir
            if (m_idx != -1 && rsb->rs[m_idx].weight != 0) {

                // Update weight and color
                float weight = rsb->rs[m_idx].weight + 1;
                if (weight > MAX_WEIGHTS)
                    rsb->rs[m_idx].weight = MAX_WEIGHTS;
                else
                    rsb->rs[m_idx].weight = weight;
                rsb->rs[m_idx].color = update_color(rsb->rs[m_idx].color, color, rsb->rs[m_idx].weight);

                // Update best color
                if (rsb->rs[m_idx].weight > info.best_weight) {
                    info.best_color = rsb->rs[m_idx].color;
                    info.best_weight = rsb->rs[m_idx].weight;
                }
            }

            // Empty slot
            else if (m_idx != -1 && rsb->rs[m_idx].weight == 0) {
                rsb->rs[m_idx].color = color;
                rsb->rs[m_idx].weight = 1;
                if (1 > info.best_weight) {
                    info.best_color = color;
                    info.best_weight = 1;
                }
            }

            // Weighted replacement
            else {
                int min_idx = info.min_idx;
                float total = info.total;

                // Randomly replace
                float randomfloat = curand_uniform(&rng_states[idx]);
                if (randomfloat * total >= rsb->rs[min_idx].weight) {
                    rsb->rs[min_idx].weight = 1;
                    rsb->rs[min_idx].color = color;
                }
            }
            
            // Motion Score (Difference)
            int diff = color_diff(color, info.best_color);
            uint8_t green = (uint8_t)(diff > 255 ? 255 : diff);
            pixel_in[x] = {color.r, green, color.b};
            //pixel_in[x] = best_color;
        }
}

void bg_estimation(ImageView<rgb8> in, dim3 grid, dim3 block) {
    
    // First frame: Allocate the reservoir buffer and the random states
    static Image<curandState> rng_states(in.width, in.height, true);
    if (rsb == nullptr)
    {
        // Reservoir init
        cudaError_t err = cudaMalloc(&rsb, in.width * in.height * sizeof(ReservoirSampleBuffer));
        if (err != cudaSuccess) {
            return;
        }
        reservoirs_init<<<grid, block>>>(rsb, in.width, in.height);

        // Random state init
        rng_states.height = in.height;
        rng_states.width = in.width;
        rng_states.stride = in.stride;
        random_state_init<<<grid, block>>>(rng_states);


    }
    if (OPTI == 0)
        bg_estimation_kernel_opti0<<<1, 1>>>(in, rsb);
    else if (OPTI == 1)
        bg_estimation_kernel_opti1<<<grid, block>>>(in, rsb, rng_states.buffer);
    else if (OPTI == 2)
        bg_estimation_kernel_opti2<<<grid, block>>>(in, rsb, rng_states.buffer);

}