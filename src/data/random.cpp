#include <torch/torch.h>
#include "data.hpp"

namespace fpath {
namespace data {
namespace datasets {
namespace {

constexpr uint32_t kTrainSize = 10;
constexpr uint32_t kWords = 1000;
constexpr uint32_t kNumChannels = 1;

// Create random text
torch::Tensor load_text() {
    auto tensor = torch::randint(
        /*low=*/0, /*high=*/kWords,
        {kTrainSize, kWords, kNumChannels}
    ); 
    
    std::cout << "Creating random text!" << std::endl;
    return tensor.to(kInt64);
}

// Create labels - all ones.
torch::Tensor load_labels() {
    auto tensor = torch::ones({kTrainSize}, torch::kInt);

    std::cout << "Creating labels!" << std::endl;
    return tensor.to(torch::kInt64);
}
} // namespace

RandomDataset::RandomDataset(Mode mode)
    : text_(load_text()),
      labels_(load_labels()) {}

torch::data::Example<> RandomDataset::get(size_t index) {
    return {text_[index], labels_[index]};
}

torch::optional<size_t> RandomDataset::size() const {
    return images_.size(0);
}

bool RandomDataset::is_train() const noexcept {
    return images_.size(0) == kTrainSize;
}

const torch::Tensor& RandomDataset::text() const {
    return text_;
}

const torch::Tensor& RandomDataset::labels() const {
    return labels_;
}
} // namespace datasets
} // namespace data
} // namespace fpath
