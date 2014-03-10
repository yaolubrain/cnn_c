#ifndef PARAMETER_H
#define PARAMETER_H
#include <vector>
#include <string>

#include "common.h"

class LayerParameter {
 public:  
  LayerParameter() {};
  ~LayerParameter() {};

  std::string type_;  
  int input_im_size_;
  int input_size_;
  int input_num_;
  int filter_size_;    
  int filter_num_; 
  int pool_dim_;
  int output_im_size_;  
  int output_size_;  
  int output_num_;  
};

class NNParameter {
 public:	
  NNParameter() {};
  ~NNParameter() {};

  int im_size_;
  int data_batch_num_;
  int data_batch_size_;   
  int sample_size_; 
  int channel_num_; 
  int class_num_;
  FileNames file_names_;
  
  int epoch_num_;  
  int mb_size_;  
  double learn_rate_;
  double momentum_;    

  std::vector<LayerParameter*> layer_para_;  
};
	
#endif