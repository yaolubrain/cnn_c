#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#define imSize 32
#define batchSize 10000
#define batchNum 5
#define channelNum 3


int main() {   
  uint8_t **images = malloc(batchNum*batchSize*sizeof(uint8_t*));
  for (int i = 0; i < batchNum*batchSize; ++i)
  	images[i] = malloc(imSize*imSize*channelNum*sizeof(uint8_t));
  uint8_t *labels = malloc(batchNum*batchSize*sizeof(uint8_t));

  char **fileNames = malloc(batchNum*sizeof(char*));
  fileNames[0] = "./data/data_batch_1.bin";
  fileNames[1] = "./data/data_batch_2.bin";
  fileNames[2] = "./data/data_batch_3.bin";
  fileNames[3] = "./data/data_batch_4.bin";
  fileNames[4] = "./data/data_batch_5.bin";  

  FILE *imFile;
  int imIdx = 0;
  for (int b = 0; b < batchNum; ++b) {
    imFile = fopen(fileNames[b], "r");
    for (int i = 0; i < batchSize; ++i) {      
      fread(&labels[imIdx], 1, 1, imFile);
      fread(images[imIdx], 1, imSize*imSize*channelNum, imFile);  
      imIdx++;
    }        
    fclose(imFile);
  }
  
  for (int i = 0; i < 2; ++i) {
    printf("%d ", labels[i]);
    for (int j = 0; j < imSize*imSize*channelNum; ++j)
      printf("%d ", images[i][j]);
  }


  
  return 0;
}
