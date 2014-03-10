#ifndef NN_H
#define NN_H
#include <iostream>
#include <vector>
#include <string>


#include "parameter.h"
#include "layer.h"
#include "common.h"

class Layer;

class NN {
 public:               
  NN(NNParameter* nn_para) {
    im_size_ = nn_para->im_size_;
    data_batch_num_ = nn_para->data_batch_num_;
    data_batch_size_ = nn_para->data_batch_size_;   
    sample_size_ = data_batch_num_ * data_batch_size_;
    channel_num_ = nn_para->channel_num_; 
    class_num_ = nn_para->class_num_;     

    epoch_num_ = nn_para->epoch_num_;  
    mb_size_ = nn_para->mb_size_;  
    learn_rate_ = nn_para->learn_rate_;
    momentum_ = nn_para->momentum_;        
    file_names_ = nn_para->file_names_;  

    nn_para_ = nn_para;    
  };

  ~NN() {
    delete[] data_im_;
    delete[] labels_;
  };

  void Init();
  void LoadData();    
  void RandPerm(int *randperm, int size);
  void Train();
  void Test();
      
  int im_size_;
  int data_batch_num_;
  int data_batch_size_;   
  int sample_size_; 
  int channel_num_; 
  int class_num_;
  int layer_num_;
  
  int epoch_num_;  
  int mb_size_;  
  double learn_rate_;
  double momentum_;    
  FileNames file_names_;

  NNParameter* nn_para_;

  double** data_im_;
  uInt8* labels_;

  std::vector<Layer*> layers_;

};


#endif