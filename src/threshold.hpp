#pragma once
#include "Image.hpp"

// Hyperparameters
#define LOW_THRESHOLD 65
#define HIGH_THRESHOLD 130
#define OPTI_THRESHOLD 1

void threshold(ImageView<rgb8> frame, ImageView<rgb8> in, dim3 grid, dim3 block);