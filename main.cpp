#include <iostream>
#include <random>
#include <cstdlib>
#include "learner.h"

#define INPUT_SIZE 4
#define OUTPUT_SIZE 16
#define DATA_SIZE 16

#define NUM_LAYER 4

#define EPOCH 300
#define MINI_BATCH_SIZE 1

#define FPATH "/Users/lunahc/Desktop/sample.txt"

using namespace std;



//
//
//  main
int main(void){
    
    int layer_size[] = {INPUT_SIZE, 20, 20, OUTPUT_SIZE};
    Net* net = new Net(layer_size, NUM_LAYER, MINI_BATCH_SIZE, EPOCH);
    
    FILE *inf = fopen(FPATH, "r");
    char input[INPUT_SIZE+1];
    char output[OUTPUT_SIZE+1];
    double d_input[DATA_SIZE][INPUT_SIZE];
    double d_output[DATA_SIZE][OUTPUT_SIZE];
    
    // file input
    for(int i=0; i<DATA_SIZE; i++){
        fscanf(inf, "%s", input);
        fscanf(inf, "%s", output);
        
        //
        //
        // string to double
        for(int j=0; j<INPUT_SIZE; j++)
            d_input[i][j] = (double)(input[j]-'0');
        for(int j=0; j<OUTPUT_SIZE; j++)
            d_output[i][j] = (double)(output[j]-'0');
    }
    
    
    // train
    net->train(DATA_SIZE, d_input, d_output);
    
    
    cout << "====================Test====================" << endl;
    // test
    for(int i=0; i<DATA_SIZE; i++){
        double* test_output;
        test_output = net->test(d_input[i]);
        
        // print test input
        for(int k=0; k<INPUT_SIZE; k++)
            cout << d_input[i][k];
        cout << endl;
        
        
        // print results
        for(int j=0; j<OUTPUT_SIZE; j++){
            cout << ((test_output[j]>=0.5)?(1):(0));
            //            cout << "[][][]ans :::" << (test_output[j]) << " ";	// precise value
        }
        cout << endl;
    }
    
    return 0;
}