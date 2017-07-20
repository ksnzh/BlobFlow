//
// Created by ksnzh on 17-7-17.
//

#include "blob.hpp"

template <typename Dtype>
void Blob<Dtype>::reshape(int num,int channels,int height,int width) {
    vector<int> shape(4);
    shape[0] = num;shape[1] = channels;
    shape[2] = height;shape[3] = width;
    reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::reshape(const BlobShape &blob_shape) {
	vector<int> shape(blob_shape.dim_size());
	for (int i = 0; i < shape.size(); ++i) {
		shape[i] = blob_shape.dim(i);
	}
	reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::reshape(vector<int> shape) {
	count_ = 1;
	shape_.resize(shape.size());
	for (int i = 0; i < shape.size(); ++i) {
		count_ *= shape[i];
		shape_[i] = shape[i];
	}
	//
	if (count_ > capacity_) {
		capacity_ = count_;
		data_.reset(new Mem(capacity_ * sizeof(Dtype)));
		diff_.reset(new Mem(capacity_ * sizeof(Dtype)));
	}
}

template <typename Dtype>
void Blob<Dtype>::reshapeLike(const Blob &blob) {
	reshape(blob.shape_);
}

template<typename Dtype>
const Dtype * Blob<Dtype>::cpu_data() const
{
	return (const Dtype*)data_->cpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype * data)
{
	return data_->set_cpu_data(data);
}

template<typename Dtype>
const Dtype * Blob<Dtype>::cpu_diff() const
{
	return (const Dtype*)diff_->cpu_data();
}

template<typename Dtype>
Dtype * Blob<Dtype>::mutable_cpu_data()
{
	return (Dtype*)data_->mutable_cpu_data();
}

template<typename Dtype>
Dtype * Blob<Dtype>::mutable_cpu_diff()
{
	return (Dtype*)diff_->mutable_cpu_data();
}

//template<typename Dtype>
//void Blob<Dtype>::update()
//{
//}
//
//template<typename Dtype>
//void Blob<Dtype>::update(int param_id)
//{
//}

template<typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool need_reshape)
{
    //copy shape
    if(need_reshape){
        vector<int> shape;
        if(proto.has_num() || proto.has_channels() || proto.has_height() || proto.has_width()){
            shape.resize(4);
            shape[0] = proto.num();shape[1] = proto.channels();
            shape[2] = proto.height();shape[3] = proto.width();
        } else{
            shape.resize(proto.shape().dim_size());
            for(int i = 0; i < proto.shape().dim_size(); ++i){
                shape[i] = proto.shape().dim(i);
            }
        }
        reshape(shape);
    }
}

template<typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff)
{
    proto->clear_shape();
    proto->clear_data();
    proto->clear_diff();
    for(int i = 0; i < shape_.size(); ++i){
        proto->mutable_shape()->add_dim(shape_[i]);
    }
    const Dtype *data = cpu_data();
    const Dtype *diff = cpu_diff();
    for(int i = 0; i < count_; ++i){
        proto->add_data(data[i]);
    }
    if(write_diff){
        for(int i = 0; i < count_; ++i){
            proto->add_diff(diff[i]);
        }
    }
}

INSTANTIATE_CLASS(Blob);
//再增加两个整型声明
template class Blob<int>;
template class Blob<unsigned int>;