#include <torch/torch.h>
#include "data.hpp"

namespace fpath {
namespace data {
namespace datasets {
namespace {

constexpr uint32_t kTrainSize = 60000;

/// Read in the Epath csv
std::tuple<torch::Tensor, torch::Tensor> read_csv(const std::str& root) {
    // TODO(Todd): read in the csv and fill out the fields in Epath::Data
    return 0
}

struct Epath::Data {
    auto text = torch::empty({kTrainSize}, torch::kInt);
    auto grade = torch::empty({kTrainSize}, torch::kInt);
}

} // namespace

/// Constructor
Epath::Epath(const std::str& root)
    : Data(read_csv(root)) {}

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
