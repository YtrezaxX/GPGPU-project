#pragma once
#include "Image.hpp"

// Hyper parameters
#define K_RESERVOIR 10
#define RGB_DIFF_THRESHOLD 15
#define MAX_WEIGHTS 200.0f
#define OPTI 2

struct ReservoirSampleInfo {
    int m_idx = -1;
    int min_idx = -1;
    float total = 0.0f;
    rgb8 best_color = {0,0,0};
    float best_weight = 0.0f;
};

struct ReservoirSample {
    rgb8 color = {0, 0, 0};
    float weight = 0.0f;
};

struct ReservoirSampleBuffer {
    ReservoirSample rs[K_RESERVOIR];
};


void bg_estimation(ImageView<rgb8> in, dim3 grid, dim3 block);
