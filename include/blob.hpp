//
// Created by ksnzh on 17-7-17.
//

#ifndef BLOBFLOW_BLOB_HPP
#define BLOBFLOW_BLOB_HPP

#include <memory>
#include <vector>
#include <string>
#include <sstream>
//#include "mem.hpp"
#include "common.hpp"
#include "protos/blobflow.pb.h"

using std::shared_ptr;
using std::vector;
using std::string;
using std::ostringstream;

//为什么是声明，而不是包含头文件
class Mem;

template <typename Dtype>
class Blob{
public:
    Blob() :data_(), diff_(), count_(), capacity_() {}
    Blob(const vector<int>& shape) :count_(0), capacity_(0) {reshape(shape_);}
    void reshape(int num, int channels, int height, int width);
    void reshape(vector<int> shape);
    void reshape(const BlobShape& blob_shape);
    void reshapeLike(const Blob& blob);
    const Dtype* cpu_data() const;
    void set_cpu_data(Dtype *data);
    const Dtype* cpu_diff() const;
    Dtype *mutable_cpu_data();
    Dtype *mutable_cpu_diff();
    ////
    //void update();
    ////
    //void update(int param_id);
    int num() const {return shape(0);}
    int channels() const {return shape(1);}
    int height() const {return shape(2);}
    int width() const {return shape(3);}
    int count() const {return count_;}
    int count(int start_axis, int end_axis) const{
        int cnt = 1;
        for(int i = start_axis; i < end_axis; ++i){
            cnt *= shape(i);
        }
        return cnt;
    }
    int count(int start_axis) const{
        return count(start_axis, num_axes());
    }
    const vector<int>& shape() const {return shape_;}
    int shape(int axis) const {return shape_[canonicalAxisIndex(axis)];}
    //debug info
    string shape_string() const {
        ostringstream stream;
		for (int i = 0; i < shape_.size(); ++i) {
			stream << shape_[i] << " ";
		}
		stream << "(" << count_ << ")";
		return stream.str();
    }
	int offset(const int n, const int c = 0, const int h = 0, const int w = 0) {
		return ((n*channels() + c) * height() + h) * width() + w;
	}
	int offset(const vector<int>& vec) {
		int offset = 0;
		for (int i = 0; i < num_axes(); ++i) {
			offset *= shape(i);
			if (vec.size() > i) {
				offset += vec[i];
			}
		}
	}
    int num_axes() const {return shape_.size();}
    //
    int canonicalAxisIndex(int axis) const{
        if(axis < 0){
            return axis + num_axes();
        }
        else{
            return axis;
        }
    }
	const shared_ptr<Mem>& data() const { return data_; }
	const shared_ptr<Mem>& diff() const { return diff_; }
	void shareData(const Blob& blob) {
		CHECK_EQ(count(), blob.count());
		data_ = blob.data();
	}
	void sharedDiff(const Blob& blob) {
		CHECK_EQ(count(), blob.count());
		diff_ = blob.diff();
	}
	void FromProto(const BlobProto& proto, bool need_reshape = true);
	void ToProto(BlobProto* proto, bool write_diff = false);
protected:
    shared_ptr<Mem> data_, diff_;
    vector<int> shape_;
    int count_, capacity_;
};

#endif //BLOBFLOW_BLOB_HPP
