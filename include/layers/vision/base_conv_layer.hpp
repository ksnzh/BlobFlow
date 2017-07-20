//
// Created by ksnzh on 17-7-19.
//

#ifndef BLOBFLOW_BASE_CONV_LAYER_HPP
#define BLOBFLOW_BASE_CONV_LAYER_HPP

#include "layer.hpp"
#include "utils/im2col.hpp"

template <typename Dtype>
class BaseConvolutionLayer :public Layer<Dtype>{
    BaseConvolutionLayer(const LayerParameter& param) :Layer<Dtype>(param) {}

};

#endif //BLOBFLOW_BASE_CONV_LAYER_HPP
