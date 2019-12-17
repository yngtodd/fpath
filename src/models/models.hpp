/// models.hpp
///
/// @file
#pragma once

#include <torch/torch.h>

namespace fpath {
namespace models {

struct YoonKim: torch::nn::Module {
    
    YoonKim(int64_t num_embeddings, int64_t embedding_dim, int64_t num_filters);

    torch::Tensor relu_pool(torch::Tensor x);

    int64_t total_filters(int64_t num_filters);

    torch::Tensor forward(const torch::Tensor& input);

} // models
} // fpath
