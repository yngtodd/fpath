// yoonkim.cpp
#include <torch/torch.h>
//#include "models.hpp"

using namespace torch;

namespace fpath {
namespace models {

/** 
 * Yoon Kim CNN for text classification
 *
 * @param num_embeddings: number of tokens to embed
 * @param embedding_dim: dimension of the word embeddings 
 * @param num_filters: number of convolution filters per layer
 */
struct YoonKim : torch::nn::Module {

    YoonKim(int64_t num_embeddings, int64_t embedding_dim, int64_t num_filters) {
        _nfilters = total_filters(num_filters);

	embed = register_module(
             "embed",
	     nn::Embedding(nn::EmbeddingOptions(
	          /*num_embeddings=*/num_embeddings,
		  /*embedding_dim=*/embedding_dim))
        );

        conv1 = register_module(
	    "conv1", 
	    nn::conv1d(nn::Conv1dOptions(
	         /*in_channels=*/embedding_dim,
		 /*out_channels=*/num_filters))
	);
        
	conv2 = register_module(
	    "conv2", 
            nn::conv1d(nn::Conv1dOptions(
	         /*in_channels=*/embedding_dim,
		 /*out_channels=*/num_filters))
	);

        conv3 = register_module(
	    "conv3", 
            nn::conv1d(nn::Conv1dOptions(
	         /*in_channels=*/embedding_dim,
		 /*out_channels=*/num_filters))
	);

	fc = register_module(
	    "fc",
	    nn::Linear(/*input_dim=*/_nfilters, /*output_dim=*/10)
	);

	// declare all the layers
	nn::Embedding embed{nullptr};
	nn::Conv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	nn::Linear fc{nullptr};
    }

    /**
     * Relu followed by max pool
     */
    torch::Tensor relu_pool(torch::Tensor x) {
         x = torch::relu(x);
	 return torch::max_pool1d(x);
    }

    /**
     * Get the total number of filters for the model
     *
     * @pram num_filters: the number of filters per convolution layer
     */
    int64_t total_filters(int64_t num_filters) {
        return num_filters * 3;
    } 

    torch::Tensor forward(const torch::Tensor& input) {
	auto embedding = embed(input);
        auto x1 = relu_pool(conv1(embedding)); 
	auto x2 = relu_pool(conv2(embedding));
	auto x3 = relu_pool(conv3(embedding));
	auto x = torch::cat({x1.view(-1, _nfilters), 
	                     x2.view(-1, _nfilters), 
	                     x3.view(-1, _nfilters)},
	                    /*dim=*/1);
	return fc(x);
    }

};
} // models
} // fpath
