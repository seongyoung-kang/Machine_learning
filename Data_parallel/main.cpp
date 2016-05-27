#include <iostream>
#include <random>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <math.h>
#include "learner.h"

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define MINI_BATCH_SIZE 8
#define EPOCH 1

/* 		Time checker		*/
struct timeval start_time, end_time, elapsed_time, exec_time;
#define START_TIME gettimeofday(&start_time, NULL);
#define END_TIME gettimeofday(&end_time, NULL);
#define PRINT_TIME timersub(&end_time, &start_time, &elapsed_time); \
printf("%ld.%d\n", exec_time.tv_sec, exec_time.tv_usec);


/*		MNIST file path		*/
#define TRAIN_PATH1 "./../Mnist/train-images.idx3-ubyte"
#define TRAIN_PATH2 "./../Mnist/train-labels.idx1-ubyte"
#define TEST_PATH1 "./../Mnist/t10k-images.idx3-ubyte"
#define TEST_PATH2 "./../Mnist/t10k-labels.idx1-ubyte"

typedef unsigned char byte;

static double True = 0;
static double False = 0;

static void train(Net* net);
static void test(Net* net);
static void error_rate(double* result, double* desired);
static void report(char* file);



//
//
//  main
int main(int ac, char* av[]){
    int layer_size[] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    Net* net = new Net(layer_size, NUM_LAYER, MINI_BATCH_SIZE, EPOCH);
    
    // train
    train(net);
    
    
    // test
    test(net);
    
    
    // report
    report(av[1]);
    
    
    return 0;
}// main





//
//
// func implements
// train
void train(Net* net){
    FILE* inf = fopen(TRAIN_PATH1, "rb");
    FILE* outf = fopen(TRAIN_PATH2, "rb");
    
    byte input[INPUT_SIZE];
    byte output[OUTPUT_SIZE];
    
    double input_double[MINI_BATCH_SIZE][INPUT_SIZE];
    double output_double[MINI_BATCH_SIZE][OUTPUT_SIZE];
    
    for(int i=0; i<EPOCH; i++){
        printf("epoch %d is running...\n", i+1);
        
        fread(input, sizeof(byte), 16, inf);	// trash
        fread(output, sizeof(byte), 8, outf);	// trash
        
        for(int j=0; j<TRAIN_SIZE/MINI_BATCH_SIZE; j++){ 		// undiviable -> cutting...
            for(int k=0; k<MINI_BATCH_SIZE; k++){
                fread(input, sizeof(byte), INPUT_SIZE, inf);
                fread(output, sizeof(byte), 1, outf);
                for(int l=0; l<INPUT_SIZE; l++)
                    input_double[k][l] = (double)input[l]!=0?1:0;
                for(int l=0; l<OUTPUT_SIZE; l++)
                    output_double[k][l] = (double)((1 << output[0]) & (1 << l))!=0?1:0;
            }
            START_TIME
            net->train(input_double, output_double, MINI_BATCH_SIZE);
            END_TIME
            timersub(&end_time, &start_time, &elapsed_time);
            timeradd(&elapsed_time, &exec_time, &exec_time);
        }
        rewind(inf);
        rewind(outf);
    }
}// train



// test
void test(Net* net){
    FILE *inf = fopen(TRAIN_PATH1, "rb");
    FILE *outf = fopen(TRAIN_PATH2, "rb");
    
    byte input[INPUT_SIZE];
    byte output[OUTPUT_SIZE];
    fread(input, sizeof(byte), 16, inf);	// trash
    fread(output, sizeof(byte), 8, outf);	// trash
    
    double input_double[INPUT_SIZE];
    double output_double[OUTPUT_SIZE];
    
    double* result;
    
    puts("Testing...");
    for(int i=0; i<TEST_SIZE; i++){
        fread(input, sizeof(byte), INPUT_SIZE, inf);
        fread(output, sizeof(byte), 1, outf);
        for(int l=0; l<INPUT_SIZE; l++)
            input_double[l] = (double)input[l]!=0?1:0;
        for(int l=0; l<OUTPUT_SIZE; l++)
            output_double[l] = (double)((1 << output[0]) & (1 << l))!=0?1:0;
        
        result = net->test(input_double);
        
        error_rate(result, output_double);
    }
}// test



void error_rate(double result[], double desired[]){
    double max = -1;
    int maxit = -1;
    
    for(int i=0; i<OUTPUT_SIZE; i++)
        if(result[i] > max){
            max = result[i];
            maxit = i;
        }
    
    for(int i=0; i<OUTPUT_SIZE; i++)
        if(i == maxit)
            result[i] = 1;
        else
            result[i] = 0;
    
    bool checker = true;
    for(int j=0; j<OUTPUT_SIZE; j++)
        if(result[j] != desired[j])
            checker = false;
    if(checker) True++;
    else False++;
}// error_rate



void report(char* file){
    puts("==========Result Report==========");
    printf("recognition rate : %lf %\n", True/(True+False)*100);
    printf("# of True : %d\n", (int)True);
    printf("# of False : %d\n", (int)False);
    printf("execution time : ");
    PRINT_TIME
    
    FILE* outf = fopen(file, "w+");
    char str[256];
    sprintf(str, "# of thread : %d\n", NUM_THREAD);
    sprintf(str, "%s mini batch size : %d\n", str, MINI_BATCH_SIZE);
    sprintf(str, "%s epoch : %d\n", str, EPOCH);
    sprintf(str, "%s learning rate : %lf\n", str, LEARNING_RATE);
    sprintf(str, "%s recognition rate : %lf %\n", str, True/(True+False)*100);
    sprintf(str, "%s execution time : %ld.%ld \n", str, exec_time.tv_sec, exec_time.tv_usec);
    fprintf(outf, "%s", str);
}// report
