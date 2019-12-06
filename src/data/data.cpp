#include <vector>
#include <tuple>
#include <torch/torch.h>
#include "data.hpp"

namespace fpath {
namespace data {
namespace datasets {
namespace {

constexpr uint32_t kTrainSize = 60000;

} // namespace

Epath::Epath(Mode mode)
    : text_(load_text()),
      targets_(load_labels()) {};

torch::data::Example<> Epath::get(size_t index) {
    return {text_[index], labels_[index]};
}

torch::optional<size_t> Epath::size() const {
    return text_.size(0);
}

bool Epath::is_train() const noexcept {
    return text_.size(0) == kTrainSize;
}

const torch::Tensor& Epath::text() const {
    return text_;
}

const torch::Tensor& Epath::targets() const {
    return targets_;
}

} // namespace datasets
} // namespace data
} // namespace fpath
