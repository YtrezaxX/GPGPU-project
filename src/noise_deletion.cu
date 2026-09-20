#include "noise_deletion.hpp"
#include <algorithm>
#include <cmath>

__device__ unsigned char atomicMinChar(unsigned char* address, unsigned char val)
{
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int sel = selectors[(size_t)address & 3];
    unsigned int old, assumed, min_, new_;

    old = *base_address;
    do {
        assumed = old;
        min_ = min(val, (unsigned char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
        new_ = __byte_perm(old, min_, sel);
        if (new_ == old)
            break;
        old = atomicCAS(base_address, assumed, new_);
    } while (assumed != old);
    return old;
}

__global__ void noise_erosion_opti0(ImageView<rgb8> in, rgb8* tmp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Extract pixels from picture
    rgb8* pixel_in = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8* pixel_tmp = (rgb8*)((std::byte*)tmp + y * in.stride);
    rgb8 color = pixel_in[x];

    if (x < in.width && y < in.height) {
        for (int y0 = y - RADIUS_EROSION; y0 < y + RADIUS_EROSION; y0++) {
            rgb8* row = (rgb8*)((std::byte*)in.buffer + y * in.stride);
            for (int x0 = x - RADIUS_EROSION; x0 < x + RADIUS_EROSION; x0++) {
                
                // Make sure box makes sense
                if (y0 >= 0 && y0 < in.height && x0 >= 0 && x0 < in.width
                && (x - x0) * (x - x0) + (y - y0) * (y - y0) <= RADIUS_EROSION * RADIUS_EROSION) {
                    if (row[x0].g < color.g) {
                        color.g = row[x0].g;
                    }    
                }                
            }
        }
        pixel_tmp[x] = color;
    }
}

__global__ void noise_diatation_opti0(ImageView<rgb8> out, rgb8* tmp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Extract pixels from picture
    rgb8* pixel_out = (rgb8*)((std::byte*)out.buffer + y * out.stride);
    rgb8* pixel_tmp = (rgb8*)((std::byte*)tmp + y * out.stride);
    rgb8 color = pixel_tmp[x];

    if (x < out.width && y < out.height) {
        for (int y0 = y - RADIUS_DILATATION; y0 < y + RADIUS_DILATATION; y0++) {
            rgb8* row = (rgb8*)((std::byte*)tmp + y * out.stride);
            for (int x0 = x - RADIUS_DILATATION; x0 < x + RADIUS_DILATATION; x0++) {
                
                // Make sure box makes sense
                if (y0 >= 0 && y0 < out.height && x0 >= 0 && x0 < out.width
                && (x - x0) * (x - x0) + (y - y0) * (y - y0) <= RADIUS_DILATATION * RADIUS_DILATATION) {
                    if (row[x0].g > color.g) {
                        color.g = row[x0].g;
                    }    
                }                
            }
        }
        pixel_out[x] = color;
    }
}

__device__ __forceinline__
int clamp(int v, int lo, int hi)
{
    return max(lo, min(v, hi));
}

// noise erosion but loading the pixels in shared memory instead of
// every thread loading pixels over and over
__global__ void noise_erosion_opti1(ImageView<rgb8> in, rgb8* out)
{
    extern __shared__ rgb8 tile[];

    const int R = RADIUS_EROSION;

    // tile dimensions including halo
    const int tile_w = blockDim.x + 2 * R;
    const int tile_h = blockDim.y + 2 * R;

    // global coordinates of output pixel
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // local coordinates inside shared tile
    int lx = threadIdx.x + R;
    int ly = threadIdx.y + R;

    // number of threads in block
    int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * blockDim.y;

    //striped arrangement loading
    int tile_size = tile_w * tile_h;

    for (int idx = tid; idx < tile_size; idx += tnum)
    {
        int ty = idx / tile_w;
        int tx = idx % tile_w;

        int gx_local = blockIdx.x * blockDim.x + tx - R;
        int gy_local = blockIdx.y * blockDim.y + ty - R;

        gx_local = clamp(gx_local, 0, in.width  - 1);
        gy_local = clamp(gy_local, 0, in.height - 1);

        tile[idx] = *((rgb8*)((uint8_t*)in.buffer + gy_local * in.stride) + gx_local);
    }

    __syncthreads();

    //out-of-bounds
    if (gx >= in.width || gy >= in.height)
        return;

    rgb8 best = tile[ly * tile_w + lx];

    //erosion
    for (int dy = -R; dy <= R; dy++)
    {
        for (int dx = -R; dx <= R; dx++)
        {
            if (dx*dx + dy*dy <= R*R)
            {
                rgb8 n = tile[(ly + dy) * tile_w + (lx + dx)];
                best.g = min(best.g, n.g);
            }
        }
    }

    // write output
    *((rgb8*)((uint8_t*)out + gy * in.stride) + gx) = best;
}

__global__ void noise_dilatation_opti1(ImageView<rgb8> in, rgb8* out)
{
    extern __shared__ rgb8 shmem[];

    const int tile_w = blockDim.x + 2 * RADIUS_DILATATION;
    const int tile_h = blockDim.y + 2 * RADIUS_DILATATION;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    int gx0 = blockIdx.x * blockDim.x - RADIUS_DILATATION;
    int gy0 = blockIdx.y * blockDim.y - RADIUS_DILATATION;

    int lx = tx + RADIUS_DILATATION;
    int ly = ty + RADIUS_DILATATION;

    for (int y = ty; y < tile_h; y += blockDim.y)
    {
        int gyy = clamp(gy0 + y, 0, in.height - 1);
        rgb8* rowIn = (rgb8*)((std::byte*)in.buffer + gyy * in.stride);

        for (int x = tx; x < tile_w; x += blockDim.x)
        {
            int gxx = clamp(gx0 + x, 0, in.width - 1);
            shmem[y * tile_w + x] = rowIn[gxx];
        }
    }

    __syncthreads();

    if (gx >= in.width || gy >= in.height)
        return;

    rgb8 color = shmem[ly * tile_w + lx];

    for (int dy = -RADIUS_DILATATION; dy <= RADIUS_DILATATION; dy++)
    {
        for (int dx = -RADIUS_DILATATION; dx <= RADIUS_DILATATION; dx++)
        {
            if (dx*dx + dy*dy <= RADIUS_DILATATION * RADIUS_DILATATION)
            {
                rgb8 n = shmem[(ly + dy) * tile_w + (lx + dx)];
                color.g = max(color.g, n.g);
            }
        }
    }

    rgb8* rowOut = (rgb8*)((std::byte*)out + gy * in.stride);
    rowOut[gx] = color;
}

void noise_deletion(ImageView<rgb8> in, dim3 grid, dim3 block) {
    // Copy for temp storing erosion
    Image<rgb8> tmp(in.width, in.height, true);

    if (OPTI_NOISE == 0) {
        // Erosion
        noise_erosion_opti0<<<grid,block>>>(in, tmp.buffer);

        // Dilatation
        noise_diatation_opti0<<<grid, block>>>(in, tmp.buffer);

    // OPTI 1
    } else if (OPTI_NOISE == 1) {

        // Erosion
        size_t shmemE = (block.x + 2*RADIUS_EROSION) * (block.y + 2*RADIUS_EROSION) * sizeof(rgb8);
        noise_erosion_opti1<<<grid,block, shmemE>>>(in, tmp.buffer);

        // Dilatation
        size_t shmemD = (block.x + 2*RADIUS_DILATATION) * (block.y + 2*RADIUS_DILATATION) * sizeof(rgb8);
        noise_dilatation_opti1<<<grid, block, shmemD>>>(in, tmp.buffer);
    }
}