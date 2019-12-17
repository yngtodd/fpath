/// Application entry point.
///
#include <chrono>
#include <vector>
#include <iostream>
#include <torch/torch.h>

#include "core/argparse.hpp"
#include "api/api.hpp"
#include "data/data.hpp"
#include "models/models.hpp"

using namespace std::chrono;

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser;
  
    parser.addArgument("-n", "--num_epochs", /*num_args=*/1, /*optional=*/false);
    parser.addArgument("-b", "--batch_size", /*num_args=*/1, /*optional=*/false);
    // parser.addArgument("-i", "--data_path", /*num_args=*/1, /*optional=*/false);
    parser.parse(argc, argv);

    int64_t num_epochs = parser.retrieve<int>("num_epochs");
    int64_t batch_size = parser.retrieve<int>("batch_size");
    // std::string data_path = parser.retrieve<std::string>("data_path");

    // Check if we can run on the GPU
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    auto dataset = fpath::data::datasets::RandomDataset()
        .map(torch::data::transforms::Stack<>());
    
    const int64_t batches_per_epoch =
        std::ceil(dataset.size().value() / static_cast<double>(batch_size));

    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
    );

    auto start = high_resolution_clock::now();

    for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
        int64_t batch_index = 0;
	for (torch::data::Example<>& batch: *data_loader) {
	    torch::Tensor img = batch.data.to(device);

	    std::printf(
                "\r[%2ld/%2ld][%3ld/%3ld]",
                epoch,
                num_epochs,
                ++batch_index,
                batches_per_epoch
	    );
	}
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "\nSummary" << std::endl;
    std::cout << "------" << std::endl;
    std::cout << "batch size: " << batch_size << std::endl;
    std::cout << "Number of epochs: " << num_epochs << std::endl;
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
