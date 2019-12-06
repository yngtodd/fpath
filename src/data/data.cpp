#include <filesystem>
#include <torch/torch.h>
#include "data.hpp"

namespace fpath {
namespace data {
namespace datasets {
namespace {

constexpr uint32_t kTrainSize = 60000;
constexpr const char* kTrainFilename = "epath_train.csv";
constexpr const char* kTestFilename = "epath_test.csv";

std::string join_paths(std::string head, const std::string& tail) {
    if (head.back() != '/') {
        head.push_back('/');
    }
    head += tail;
    return head;
}

/// Read in the Epath csv
std::tuple<torch::Tensor, torch::Tensor> load_text(const std::string& root, bool train) {
    const auto num_samples = train ? kTrainSize : kTestSize;

    const auto path =
        join_paths(root, train ? kTrainFilename : kTestFilename); 

    std::ifstream data(path);

    // return
}

} // namespace

/// Constructor
Epath::Epath(const std::str& root)
    : Epath{read_csv(root)} {}

/// Get a batch of data at an index
torch::data::Example<> Epath::get(size_t index) {
    return {text_[index], targets_[index]};
}

/// Get the size of the dataset
torch::optional<size_t> Epath::size() const {
    return text_.size(0);
}

/// Check whether this is the train set
bool Epath::is_train() const noexcept {
    return text_.size(0) == kTrainSize;
}

/// Return all of the text data as a single Pytorch tensor
const torch::Tensor& Epath::text() const {
    return text_;
}

/// Return all of the labels as a single Pytorch tensor
const torch::Tensor& Epath::targets() const {
    return targets_;
}

} // namespace datasets
} // namespace data
} // namespace fpath
