#include <iostream>
#include <random>
#include <fstream>
#include <cstdlib>
#include "learner.h"

#define TRAIN_SIZE 16
#define TEST_SIZE 16

#define MINI_BATCH_SIZE 1
#define EPOCH 100


#define TRAIN_PATH1 "/Users/lunahc/Desktop/Mnist/trainer1.txt"
#define TRAIN_PATH2 "/Users/lunahc/Desktop/Mnist/trainer2.txt"
#define TEST_PATH1 "/Users/lunahc/Desktop/Mnist/tester1.txt"
#define TEST_PATH2 "/Users/lunahc/Desktop/Mnist/tester2.txt"

/*		MNIST		*/
//#define TRAIN_PATH1 "/Users/lunahc/Desktop/Mnist/train-images.idx3-ubyte"
//#define TRAIN_PATH2 "/Users/lunahc/Desktop/Mnist/train-labels.idx1-ubyte"
//#define TEST_PATH1 "/Users/lunahc/Desktop/Mnist/t10k-images.idx3-ubyte"
//#define TEST_PATH2 "/Users/lunahc/Desktop/Mnist/t10k-labels.idx1-ubyte"

typedef unsigned char byte;

static void train(Net* net);
static void test(Net* net);



//
//
//  main
int main(void){
    int layer_size[] = {INPUT_SIZE, HIDDEN_SIZE , OUTPUT_SIZE};
    Net* net = new Net(layer_size, NUM_LAYER, MINI_BATCH_SIZE, EPOCH);
   

    // train
    train(net);

    
    // test
    test(net);
   

    delete net;
    return 0;
}// main




//
//
// func implements
// train
void train(Net* net){
    ifstream inf(TRAIN_PATH1);
    ifstream outf(TRAIN_PATH2);
    
    byte input_str[INPUT_SIZE+1];
    byte output_str[OUTPUT_SIZE+1];
    double input[MINI_BATCH_SIZE][INPUT_SIZE];
    double output[MINI_BATCH_SIZE][OUTPUT_SIZE];
    
    for(int l=0; l<EPOCH; l++){
        for(int k=0; k<TRAIN_SIZE/MINI_BATCH_SIZE; k++){ 		// undiviable -> cutting...
            for(int j=0; j<MINI_BATCH_SIZE; j++){
                inf >> input_str;
                outf >> output_str;
               
                for(int i=0; i<INPUT_SIZE; i++)
                    input[j][i] = (double)input_str[i]-'0';
                for(int i=0; i<OUTPUT_SIZE; i++)
                    output[j][i] = (double)output_str[i]-'0';
            }
            net->train(input, output, MINI_BATCH_SIZE);
        }
        inf.clear();
        outf.clear();
        inf.seekg(0, ifstream::beg);
        outf.seekg(0, ifstream::beg);
    }
}// train







// test
void test(Net* net){
    FILE *inf = fopen(TEST_PATH1, "r");
    FILE *outf = fopen(TEST_PATH2, "r");
    
    byte input_str[INPUT_SIZE+1];
    byte output_str[OUTPUT_SIZE+1];
    
    double input[INPUT_SIZE];
    double output[OUTPUT_SIZE];
    
    double* result;
    
    for(int i=0; i<TEST_SIZE; i++){
        
        fscanf(inf, "%s", input_str);
        fscanf(outf, "%s", output_str);
        
        for(int j=0; j<INPUT_SIZE; j++)
            input[j] = (double)input_str[j]-'0';
        for(int j=0; j<OUTPUT_SIZE; j++)
            output[j] = (double)output_str[j]-'0';
        
        result = net->test(input);
        
        
        /********************************* print result *****************************/
        std::cout << "=====INPUT=====" << std::endl;
        for(int j=0; j<INPUT_SIZE; j++)
            std::cout << input[j];
        std::cout << std::endl;
        std::cout << "=====OUTPUT=====" << std::endl;
        for(int j=0; j<OUTPUT_SIZE; j++)
            std::cout << (result[j]>=0.5?1:0);
//            std::cout << result[j];
        std::cout << std::endl;
        std::cout << "=====ANS=====" << std::endl;
        for(int j=0; j<OUTPUT_SIZE; j++)
            std::cout << output[j] << " ";
        std::cout << std::endl;
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    }
}// test
