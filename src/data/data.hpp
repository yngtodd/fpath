// data.hpp
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/torch.h>

namespace fpath {
namespace data {
namespace datasets {

/// Epath dataset
///
class Epath : public torch::data::Dataset<Epath>
    public:
        /// The mode in which the dataset is loaded.
        enum class Mode { kTrain, kTest };

        explicit Epath(Mode mode = Mode::kTrain);

        /// Returns the `Example` at the given `index`.
        torch::data::Example<> get(size_t index) override;

        /// Returns the size of the dataset.
        torch::optional<size_t> size() const override;

        /// Returns true if this is the training subset of MNIST.
        bool is_train() const noexcept;

        /// Returns all images stacked into a single tensor.
        const torch::Tensor& text() const;

        /// Returns all targets stacked into a single tensor.
        const torch::Tensor& targets() const;

    private:
        torch::Tensor text_, targets_;
};
} // namespace datasets
} // namespace data
} // namespace fpath
