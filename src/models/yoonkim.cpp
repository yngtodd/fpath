// yoonkim.cpp
#include <torch/torch.h>
#include "models.hpp"

using namespace torch;

namespace fpath {
namespace models {

/// Yoon Kim CNN for text classification
/// 
struct YoonKim : torch::nn::Module {

    YoonKim() {
	embed = register_module(
             "embed",
	     nn::Embedding(nn::EmbeddingOptions(
	          /*num_embeddings=*/1000,
		  /*embedding_dim=*/300))
        );

        conv1 = register_module(
	    "conv1", 
	    nn::conv1d(nn::Conv1dOptions(
	         /*in_channels=*/300,
		 /*out_channels=*/3))
	);
        
	conv2 = register_module(
	    "conv2", 
            nn::conv1d(nn::Conv1dOptions(
	         /*in_channels=*/300,
		 /*out_channels=*/3))
	);

        conv3 = register_module(
	    "conv3", 
            nn::conv1d(nn::Conv1dOptions(
	         /*in_channels=*/300,
		 /*out_channels=*/3))
	);

	fc = register_module(
	    "fc",
	    nn::Linear(/*input_dim=*/0, /*output_dim=*/0)
	);
    }

    torch::Tensor relu_pool(torch::Tensor x) {
         x = torch::relu(x);
	 return torch::max_pool1d(x);
    }

    torch::Tensor forward(const torch::Tensor& input) {
	auto embedding = embed(input);
        auto x1 = relu_pool(conv1(embedding)); 
	auto x2 = relu_pool(conv2(embedding));
	auto x3 = relu_pool(conv3(embedding));
	auto x = torch::cat([x1, x2, x3], /*dim=*/0);
	return fc(x);
    }

} // models
} // fpath
