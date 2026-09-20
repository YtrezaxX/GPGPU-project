#include "threshold.hpp"

__global__ void init_images(ImageView<rgb8> in, bool* input, bool* marker, bool* out, int bool_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Row of mask
    rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    bool* pixel_input = (bool*)((std::byte*)input + y * bool_stride);
    bool* pixel_marker = (bool*)((std::byte*)marker + y * bool_stride);
    bool* pixel_out = (bool*)((std::byte*)out + y * bool_stride);

    if (x < in.width && y < in.height) {

        // Put in pixel to false if in < Low Threshold
        pixel_input[x] = pixel_in[x].g >= LOW_THRESHOLD;

        // Put marker pixel to false if marker < High Threshold
        pixel_marker[x] = pixel_in[x].g >= HIGH_THRESHOLD;

        // Put out pixel to false
        bool i_got_accepted_at_nvidia = false;
        pixel_out[x] = i_got_accepted_at_nvidia;
    }
}

__device__ bool neighbors_active(bool* marker, int x, int y, int stride) {
    // Check 8 neighbors
    bool* row_above = (bool*)((std::byte*)marker + (y - 1) * stride);
    bool* row_current = (bool*)((std::byte*)marker + y * stride);
    bool* row_below = (bool*)((std::byte*)marker + (y + 1) * stride);

    if (row_above[x - 1]) return true;
    if (row_above[x])     return true;
    if (row_above[x + 1]) return true;
    if (row_current[x - 1]) return true;
    if (row_current[x + 1]) return true;
    if (row_below[x - 1]) return true;
    if (row_below[x])     return true;
    if (row_below[x + 1]) return true;

    return false;
}


__device__ bool has_changed = false;

__global__ void reconstruction_opti0(bool* input, bool* marker, bool* output, int width, int height, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Row of image
    bool* pixel_input = (bool*)((std::byte*)input + y * stride);
    bool* pixel_marker = (bool*)((std::byte*)marker + y * stride);
    bool* pixel_output = (bool*)((std::byte*)output + y * stride);


    // Already processed or too low
    if (pixel_output[x] || !pixel_input[x]) // already processed or too low
       return;

    // Init output with markers
    if (pixel_marker[x]) {
        pixel_output[x] = true;
        has_changed = true;
        return;
    }

    // Check if neighbors active
    if (neighbors_active(marker, x, y, stride)) {
        pixel_output[x] = true;
        has_changed = true;
    }
}

__global__ void apply_mask(ImageView<rgb8> in, bool* mask, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Row of image
    rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    bool* pixel_mask = (bool*)((std::byte*)mask + y * stride);

    // If in range put pixel g value to 255 if in mask else 0
    if (x < in.width && y < in.height) {
        
        if (pixel_mask[x])
            pixel_in[x].g = min(255, pixel_in[x].g + 125);
    }
}

// OPTI 1 --------------------------------------------------------------------

__device__ int change_counter = 1;

__global__ void init_mask(ImageView<rgb8> in, char* masks, int byte_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Row of mask
    rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    char* pixel_mask =((char*)masks + y * byte_stride);

    if (x < in.width && y < in.height) {
        char value = char(0);

        // Set input bit if in pixel >= Low Threshold
        if (pixel_in[x].g >= LOW_THRESHOLD) {
            value |= (1u << 0);
        }
        // Leave second bit for pixel activation at step before


        // Set out bit if in pixel >= High Threshold
        if (pixel_in[x].g >= HIGH_THRESHOLD) {
            value |= (1u << 2);
        }
        pixel_mask[x] = value;
    }
}

__global__ void reconstruction_opti1(char* masks, int width, int height, int stride, bool switchIndex) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // Row of image
    char* pixel_masks = (char*)((std::byte*)masks + y * stride);

    // Indices to check whether pixel was active in previous step
    int i1 = switchIndex ? 3 : 1;
    int i2 = switchIndex ? 1 : 3;

    // Reset other side's active flag
    pixel_masks[x] &= ~(1u << i2);


    // Already processed or too low
    if (!(pixel_masks[x] & (1u << i1))) // already processed or too low
       return;

    // Activate inactive neighbors
    bool activated = false;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {

            // Skip self
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                char* neighbor_pixel = (char*)((std::byte*)masks + ny * stride);

                // If neighbor is input active but not output active
                if ((neighbor_pixel[nx] & (1u << 0)) && !(neighbor_pixel[nx] & (1u << 2))) {

                    // Add to output
                    neighbor_pixel[nx] |= (1u << 2);

                    // Mark as activated at last step
                    neighbor_pixel[nx] |= (1u << i2);
                    activated = true;
                }
            }
        }
    }
    if (activated) {
        atomicAdd(&change_counter, 1);
    }
}

__global__ void apply_mask_opti1(ImageView<rgb8> in, char* masks, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Row of image
    rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    char* pixel_mask = (char*)((std::byte*)masks + y * stride);

    // If in range put pixel g value to 255 if in mask else 0
    if (x < in.width && y < in.height) {
        
        if (pixel_mask[x] & (1u << 2))
            pixel_in[x].g = min(255, pixel_in[x].g + 125);
    }
}

void threshold(ImageView<rgb8> frame, ImageView<rgb8> in, dim3 grid, dim3 block) {

    if (OPTI_THRESHOLD == 0) {

        //Init input, marker and out masks
        Image<bool> input(in.width, in.height, true);
        Image<bool> marker(in.width, in.height, true);
        Image<bool> output(in.width, in.height, true);
        init_images<<<grid, block>>>(in, input.buffer, marker.buffer, output.buffer, input.stride);

        // Proceed to threshold until it stops moving (creating duplicates changed because cant rread from device)
        bool changed = false;
        do {
            // Put has_changed to false
            changed = false;
            cudaMemcpyToSymbol(has_changed, &changed, sizeof(bool));

            // Execute kernel
            reconstruction_opti0<<<grid, block>>>(input.buffer, marker.buffer, output.buffer, in.width, in.height, input.stride);

            // Copy back has_changed to host
            cudaMemcpyFromSymbol(&changed, has_changed, sizeof(bool)); // read dev flag

        } while(changed);

        //  Change in to reflect mask
        cudaMemcpy2D(in.buffer, in.stride, frame.buffer, frame.stride, frame.width * sizeof(rgb8), frame.height, cudaMemcpyHostToDevice);
        apply_mask<<<grid, block>>>(in, output.buffer, output.stride);
    }
    else if (OPTI_THRESHOLD == 1) {

        // only one mask to hold all data
        Image<char> masks(in.width, in.height, true);
        init_mask<<<grid, block>>>(in, masks.buffer, masks.stride);

        int counter = 2;
        bool switchIndex = false;
        while (counter != 0)
        {
            // Reset counter
            counter = 0;
            cudaMemcpyToSymbol(change_counter, &counter, sizeof(int));

            // Execute kernel
            reconstruction_opti1<<<grid, block>>>(masks.buffer, in.width, in.height, masks.stride, switchIndex);

            // Copy back counter to host
            cudaMemcpyFromSymbol(&counter, change_counter, sizeof(int));
            switchIndex = !switchIndex;
        }
        //  Change in to reflect mask
        cudaMemcpy2D(in.buffer, in.stride, frame.buffer, frame.stride, frame.width * sizeof(rgb8), frame.height, cudaMemcpyHostToDevice);
        apply_mask_opti1<<<grid, block>>>(in, masks.buffer, masks.stride);
    }
}