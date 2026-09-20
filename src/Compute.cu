#include "Compute.hpp"
#include "bg_estimation.hpp"
#include "noise_deletion.hpp"
#include "threshold.hpp"
#include "logo.h"
#include <algorithm>
#include <vector>
#include <iterator>

// Single threaded version of the Method
__global__ void mykernel(ImageView<rgb8> in, ImageView<uint8_t> logo)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height)
    {
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        pixel[x].r = 255;

        if (x < logo.width && y < logo.height)
        {
            float alpha = logo.buffer[y * logo.stride + x] / 255.f;
            pixel[x].g = uint8_t(alpha * pixel[x].g + (1 - alpha) * 255);
            pixel[x].b = uint8_t(alpha * pixel[x].b + (1 - alpha) * 255);
        }
    }
}


void compute_cu(ImageView<rgb8> frame)
{
    static Image<uint8_t> device_logo;

    dim3 block(16, 16);
    dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

    // Set correct GPU
    cudaSetDevice(0);
    
    // Copy the logo to the device if it is not already there
    if (device_logo.buffer == nullptr)
    {
        device_logo = Image<uint8_t>(logo_width, logo_height, true);
        cudaMemcpy2D(device_logo.buffer, device_logo.stride, logo_data, logo_width, logo_width, logo_height, cudaMemcpyHostToDevice);
    }

    // Copy the input image to the device
    static Image<rgb8> in;
    if (in.width != frame.width || in.height != frame.height)
        in = Image<rgb8>(frame.width, frame.height, true);
    cudaMemcpy2D(in.buffer, in.stride, frame.buffer, frame.stride, frame.width * sizeof(rgb8), frame.height, cudaMemcpyHostToDevice);
    
    //Logo mykernel<<<grid, block>>>(device_in, device_logo);

    // Init background estimation pointer and reservoirs
    bg_estimation(in, grid, block);
    //cudaMemcpy2D(frame.buffer, frame.stride, in.buffer, in.stride, frame.width * sizeof(rgb8), frame.height, cudaMemcpyDeviceToHost);
    //return;

    // Erosion + dilatation
    noise_deletion(in, grid, block);

    // Hysteresis threshold
    threshold(frame, in, grid, block);

    cudaDeviceSynchronize();
    // Copy the result back to the host
    cudaMemcpy2D(frame.buffer, frame.stride, in.buffer, in.stride, frame.width * sizeof(rgb8), frame.height, cudaMemcpyDeviceToHost);
    //TODO: Free at the poiter at the end cudaFree(d_reservoirs);
}