#include <torch/extension.h>

torch::Tensor flash_attention_1_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
std::vector<torch::Tensor> flash_attention_2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v1", torch::wrap_pybind_function(flash_attention_1_forward), "flash_attention_1_forward");
    m.def("forward_v2", torch::wrap_pybind_function(flash_attention_2_forward), "flash_attention_2_forward");
}

