#include <iostream>
#include <random>
#include <cstdlib>
#include "learner.h"

#define NUM_LAYER 4

#define INPUT_SIZE 4
#define HIDDEN_SIZE 20, 20
#define OUTPUT_SIZE 16

#define TEST_SIZE 16
#define MINI_BATCH_SIZE 1
#define EPOCH 300

#define DATA_SIZE 16

#define TRAIN_PATH "/Users/lunahc/Desktop/trainer.txt"
#define TEST_PATH "/Users/lunahc/Desktop/tester.txt"

static double* str_to_double(char* str, double* output, int size);
static void read_train(double input[][INPUT_SIZE], double output[][OUTPUT_SIZE]);
static void read_test(double data[][INPUT_SIZE]);
static void test(Net* net, double input[][INPUT_SIZE]);



//
//
//  main
int main(void){
    int layer_size[] = {INPUT_SIZE, HIDDEN_SIZE , OUTPUT_SIZE};
    Net* net = new Net(layer_size, NUM_LAYER, MINI_BATCH_SIZE, EPOCH);
    
    double input[DATA_SIZE][INPUT_SIZE];
    double output[DATA_SIZE][OUTPUT_SIZE];
    
    double test_input[TEST_SIZE][INPUT_SIZE];
    
    // read input & output data on file
    read_train(input, output);

    
    // train with data
    net->train(DATA_SIZE, input, output);
    
    
    // read test data on file
    read_test(test_input);
   
    
    // test
    test(net, test_input);
    
    
    delete net;
    return 0;
}// main



//
//
// func implements
double* str_to_double(char* str, double* output, int size){
    for(int i=0; i<size; i++)
        output[i] = (double)(str[i]-'0');
    return output;
}// str_to_double

void read_train(double input[][INPUT_SIZE], double output[][OUTPUT_SIZE]){
    FILE *inf = fopen(TRAIN_PATH, "r");
    char input_str[INPUT_SIZE+1];
    char output_str[OUTPUT_SIZE+1];
    
    for(int i=0; i<DATA_SIZE; i++){
        fscanf(inf, "%s", input_str);
        fscanf(inf, "%s", output_str);
        
        str_to_double(input_str, input[i], INPUT_SIZE);
        str_to_double(output_str, output[i], OUTPUT_SIZE);
    }
}// read_train

void read_test(double data[][INPUT_SIZE]){
    FILE *inf = fopen(TEST_PATH, "r");
    char test_str[TEST_SIZE+1];
    
    for(int i=0; i<TEST_SIZE; i++){
        fscanf(inf, "%s", test_str);
        
        str_to_double(test_str, data[i], INPUT_SIZE);
    }
}// read_test

void test(Net* net, double input[][INPUT_SIZE]){
    double* output;
    for(int i=0; i<TEST_SIZE; i++){
        std::cout << " TEST DATA : ";
        for(int j=0; j<INPUT_SIZE; j++)
            std::cout << input[i][j];
        std::cout << std::endl;
        
        output = net->test(input[i]);
        
        std::cout << " RESULT    : ";
        for(int j=0; j<OUTPUT_SIZE; j++)
            std::cout << ((output[j]>=0.5)?1:0);
        std::cout << std::endl;
    }
}// test