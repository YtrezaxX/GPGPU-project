# pragma once
# include "Image.hpp"

// Hyper parameters
#define RADIUS_EROSION 4
#define RADIUS_DILATATION 4
#define OPTI_NOISE 1


void noise_deletion(ImageView<rgb8> in, dim3 grid, dim3 block);