#include <torch/torch.h>
#include "data.hpp"

namespace fpath {
namespace data {
namespace datasets {
namespace {

constexpr uint32_t kTrainSize = 60000;

/// Read in the Epath csv
Epath::Data read_csv(const std::string& root, Mode mode) {
    // TODO(Todd): read in the csv and fill out the fields in Epath::Data
    return 0;
}
} // namespace

Epath::Data {
    torch::Tensor text;
    torch::Tensor grade;
}

/// Constructor
Epath::Epath(const std::string& root, Mode mode)
    : Data(read_csv(root, mode)) {}

/// Get a batch of data at an index
torch::data::Example<> Epath::get(size_t index) {
    return {Data.text[index], Data.grade[index]};
}

/// Get the size of the dataset
torch::optional<size_t> Epath::size() const {
    return Data.text.size(0);
}

/// Check whether this is the train set
bool Epath::is_train() const noexcept {
    return Data.text.size(0) == kTrainSize;
}

/// Return all of the text data as a single Pytorch tensor
const torch::Tensor& Epath::text() const {
    return Data;
}

} // namespace datasets
} // namespace data
} // namespace fpath
