#include <iostream>
#include <random>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include "learner.h"

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define MINI_BATCH_SIZE 1
#define EPOCH 1

#define MAX_FILE_NAME 20

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
static void report(int nThread, char* file);



//
//
//  main
int main(int ac, char* av[]){
    int nThread;
    char resultFileName[MAX_FILE_NAME];

    if(ac > 3){
        fprintf(stderr, "usage: %s {#Thread} {fileName}", av[0]);
        goto leave;
    }

    (ac == 1)? nThread = 1: nThread = atoi(av[1]);
    (ac == 3)? strcat(resultFileName, av[2]): strcat(resultFileName, "result.txt");

    int layer_size[] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    Net* net = new Net(layer_size, NUM_LAYER, MINI_BATCH_SIZE, EPOCH, nThread);
    
    // train
    train(net);
    
    // test
    test(net);

    report(nThread, resultFileName);

leave:
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
    
    for(int i=0; i<OUTPUT_SIZE; ++i)
        if(result[i] > max)
            max = result[i], maxit = i;
    
    for(int i=0; i<OUTPUT_SIZE; ++i)
        (i == maxit)? result[i] = 1: result[i] = 0;
    
    bool checker = true;
    for(int j=0; j<OUTPUT_SIZE; j++)
        if(result[j] != desired[j])
            checker = false;
    (checker)? True++: False++;
}// error_rate



void report(int nThread, char* file){
    int i;
    double recogRate = True*100/(True+False);

    /* Print Report */
    puts("==========Result Report==========");
    printf("# of thread : %d\n", nThread);
    printf("Minibatch size : %d\n", MINI_BATCH_SIZE);
    printf("Epoch : %d\n", EPOCH);
    printf("Learning rate : %d\n", LEARNING_RATE);
    printf("# of correct answer: %d\n", (int)True);
    printf("# of incorrect answer : %d\n", (int)False);
    printf("Recognition rate : %lf %\n", recogRate);
    printf("Execution time : ");
    PRINT_TIME
    for(i=0; i<3; i++)
        printf("section [%d] : %ld.%d\n", i, sum_t[i].tv_sec, sum_t[i].tv_usec);
    
    /* Make report file */
    FILE* outf = fopen(file, "w+");
    char str[256];
    sprintf(str, "# of thread : %d\n", nThread);
    sprintf(str, "%s Mini batch size : %d\n", str, MINI_BATCH_SIZE);
    sprintf(str, "%s Epoch : %d\n", str, EPOCH);
    sprintf(str, "%s Learning rate : %lf\n", str, LEARNING_RATE);
    sprintf(str, "%s Recognition rate : %lf %\n", str, recogRate);
    sprintf(str, "%s Execution time : %ld.%ld\n", str, exec_time.tv_sec, exec_time.tv_usec);
    for(i=0; i<3; i++)
        sprintf(str, "%s section [%d] : %ld.%d\n", str, i, sum_t[i].tv_sec, sum_t[i].tv_usec);
    fprintf(outf, "%s", str);
}// report
