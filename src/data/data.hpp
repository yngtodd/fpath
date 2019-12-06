// data.hpp
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/torch.h>

namespace fpath {
namespace data {
namespace datasets {
namespace {

/// Read in the Epath csv
std::tuple<torch::Tensor, torch::Tensor> read_csv(const std::str root);
} // namespace

/// Epath dataset
///
class Epath : public torch::data::Dataset<Epath>
    public:
        Epath(const std::str& root) : Epath{read_csv(root)}

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
	Epath(std::tuple<torch::Tensor, torch::Tensor> t)
            : text_{std::get<0>(t)},
	      targets_{std::get<1>(t)};
};
} // namespace datasets
} // namespace data
} // namespace fpath
