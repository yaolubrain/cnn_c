#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cblas.h>

#include "nn.h"
#include "layer.h"
#include "common.h"

void NN::Init() {     
  for (int i = 0; i < nn_para_->layer_para_.size(); ++i) {
    Layer* new_layer = GetLayer(nn_para_, nn_para_->layer_para_[i]);    
    layers_.push_back(new_layer);    
  }

  for (int i = 0; i < nn_para_->layer_para_.size(); ++i) {
    if (i == 0) {    
      layers_[i]->Init(NULL, layers_[i+1]);  
    } else if (i == nn_para_->layer_para_.size() - 1) {
      layers_[i]->Init(layers_[i-1], NULL);  
    } else {      
      layers_[i]->Init(layers_[i-1], layers_[i+1]);  
    }
  }

  std::cout << "Network Initialization succeeded" << std::endl;
}

void NN::LoadData() {     
  data_im_ = new double* [sample_size_];
  for (int i = 0; i < sample_size_; ++i) {
      data_im_[i] = new double [im_size_*im_size_];
  }
  labels_ = new uInt8[sample_size_];
  
  uInt8 **data_im_RGB = new uInt8* [sample_size_];  
  for (int i = 0; i < sample_size_; ++i) {
    data_im_RGB[i] = new uInt8 [im_size_*im_size_*channel_num_];
  }

  std::ifstream data_file;
  
  int im_idx = 0;
  for (int i = 0; i < data_batch_num_; ++i) {
    data_file.open(file_names_[i].c_str(), std::ios::binary);
    for (int j = 0; j < data_batch_size_; ++j) {
      data_file.read((char*) &labels_[im_idx], 1);
      data_file.read((char*) data_im_RGB[im_idx], im_size_*im_size_*channel_num_);
      ++im_idx;
    }
    data_file.close();
  }
  
  // RGB to grayscale
  for (int i = 0; i < sample_size_; ++i) {
    for (int j = 0; j < im_size_*im_size_; ++j) {
      data_im_[i][j] = (double) 0.2989*data_im_RGB[i][j] 
                + 0.5870*data_im_RGB[i][j+im_size_*im_size_]
                + 0.1140*data_im_RGB[i][j+2*im_size_*im_size_]; 
    }    
  }

  for (int i = 0; i < sample_size_; ++i) {
    delete[] data_im_RGB[i];
  }
  delete[] data_im_RGB;
  
  std::cout << "Data loading succeeded" << std::endl;
}

void NN::RandPerm(int *randperm, int size) {  
  for (int i = 0; i < size; ++i) {
    randperm[i] = i;
  }
  for (int i = 0; i < size; ++i) {
    int j = rand()%(size - i) + i;
    int t = randperm[j];
    randperm[j] = randperm[i];
    randperm[i] = t;
  }
}


void NN::Train() {
  for (int e = 0; e < epoch_num_; ++e) {
    for (int i = 0; i < sample_size_; ++i) {                    

      layers_[0]->input_ = data_im_[i];
      for (int j = 0; j < layers_.size(); ++j) {        
        layers_[j]->Forward();               
      }
        
      layers_[layers_.size() - 1]->label_ = labels_[i];
      for (int j = layers_.size() - 1; j >= 0; --j) {
        layers_[j]->Backward();            
      }

      for (int j = 0; j < layers_.size(); ++j) {   
        layers_[j]->GetGradient();                             
      }
      
      if (i % nn_para_->mb_size_ == 0) {
        for (int j = 0; j < layers_.size(); ++j) {   
          layers_[j]->Update();               
        }
        std::cout << dynamic_cast<FullLayer*>(layers_[4])->bias_[0] << std::endl;      
      }      
    }

    std::cout << "train!" << std::endl;
  }  
}



