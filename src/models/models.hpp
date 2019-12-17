/// models.hpp
///
/// @file
#pragma once

#include <torch/torch.h>

namespace fpath {
namespace models {

struct YoonKim: torch::nn::Module {

    YoonKim(int64_t num_embeddings, int64_t embedding_dim, int64_t num_filters);

    // ReLu => max pool 1D
    torch::Tensor relu_pool(torch::Tensor x);

    torch::Tensor forward(const torch::Tensor& input);

    // Total number of filters for the model
    int64_t total_filters(int64_t num_filters);
};

} // models
} // fpath
