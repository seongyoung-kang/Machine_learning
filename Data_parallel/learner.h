#ifndef learner_h
#define learner_h

#include <omp.h>

#define NUM_LAYER 7

#define INPUT_SIZE 784
#define HIDDEN_SIZE 800, 600, 400, 200, 100
#define OUTPUT_SIZE 10

#define LEARNING_RATE 0.0005

#define NUM_THREAD 8

/******************************************************
 *
 * num_layer : number of layers (include input, output layer)
 *
 * layer_size : array for each layers size
 *
 * value : each node's output
 *
 ******************************************************/

class Net{
private:
    int nCPU;
    int num_layer;
    int mini_batch_size;
    int* layer_size;
    double*** value;
    double*** weight;
    double** bias;
    double*** error;
    
public:
    Net(int* layer_size, int num_layer, int mini_batch_size, int epoch);
    ~Net();
    
    void train(double input[][INPUT_SIZE], double desired[][OUTPUT_SIZE], int num_data);
    double* test(double* input);
    
private:
    void initializer();
    
    double sigmoid(double num);
    
    void feedforward(double* input, int data_num);
    void back_pass(double* desired, double** error, int data_num);
    void backpropagation(double learning_rate, int num_data);
    
};// Net class





//
//
//		implementations

// constructor
Net::Net(int* layer_size, int num_layer, int mini_batch_size, int epoch){
    nCPU = NUM_THREAD;
    
    int i, j, k;
    //
    // write on class vars
    this->num_layer = num_layer;
    this->mini_batch_size = mini_batch_size;
    
    
    //
    //          memory allocation
    // layer size
    this->layer_size = new int[num_layer];
    for(i=0; i<num_layer; i++)
        this->layer_size[i] = layer_size[i];
    
    // value
    value = new double**[mini_batch_size];
    for(i=0; i<mini_batch_size; i++)
        value[i] = new double*[num_layer];
    for(i=0; i<mini_batch_size; i++)
        for(j=0; j<num_layer; j++)
            value[i][j] = new double[layer_size[j]];
    
    // error
    error = new double**[mini_batch_size];
    for(i=0; i<mini_batch_size; i++)
        error[i] = new double*[num_layer];
    for(i=0; i<mini_batch_size; i++)
        for(int j=0; j<num_layer; j++)
            error[i][j] = new double[layer_size[j]];
    
    // bias
    bias = new double*[num_layer];
    for(i=1; i<num_layer; i++)
        bias[i] = new double[layer_size[i]];  //  bias[0] : trash
    
    // weight
    weight = new double**[num_layer-1];
    for(i=0; i<num_layer-1; i++)
        weight[i] = new double*[layer_size[i]];
    for(i=0; i<num_layer-1; i++)
        for(j=0; j<layer_size[i]; j++)
            weight[i][j] = new double[layer_size[i+1]];
    
    //
    //
    //          input random initial weights & biases(-1 ~ 1)
    for(i=0; i<num_layer-1; i++)
        for(j=0; j<layer_size[i]; j++)
            for(k=0; k<layer_size[i+1]; k++)
                weight[i][j][k] = (double)rand()/(RAND_MAX/2)-1;
    
    for(i=1; i<num_layer; i++)
        for(j=0; j<layer_size[i]; j++)
            bias[i][j] = 0;
    
}// constructor



void Net::train(double input[][INPUT_SIZE], double desired[][OUTPUT_SIZE], int num_data){
    int i;
    initializer();
#pragma omp parallel for num_threads(num_data)
    for(i=0; i<num_data; i++){
        feedforward(input[i], i);
        back_pass(desired[i], error[i], i);
    }
    backpropagation(LEARNING_RATE, num_data);
}// train



double* Net::test(double* input){
    initializer();
    feedforward(input, 0);
    return value[0][num_layer-1];
}// test



void Net::initializer(){
    int i, j, k;
#pragma omp parallel for num_threads(nCPU)
    for(k=0; k<mini_batch_size; k++)
        for(i=1; i<num_layer; i++)
            for(j=0; j<layer_size[i]; j++)
                error[k][i][j] = 0;
}// initializer



double Net::sigmoid(double num){
    return 1/(1 + exp(-num));
}// sigmoid



void Net::feedforward(double* input, int data_num){
    int i, j, k;
    for(i=0; i<layer_size[0]; i++)
        value[data_num][0][i] = input[i];
    
    for(i=0; i<num_layer-1; i++)
        for(k=0; k<layer_size[i+1]; k++){
            double sum = 0;
            for(j=0; j<layer_size[i]; j++)
                sum += weight[i][j][k]*value[data_num][i][j];
            value[data_num][i+1][k] = sigmoid(sum + bias[i+1][k]);
        }
}// feed forward



void Net::back_pass(double* desired, double** error, int data_num){
    int i, j, k;
    for(i=0; i<layer_size[num_layer-1]; i++)
        error[num_layer-1][i] = value[data_num][num_layer-1][i] - desired[i];
    
    for(i=num_layer-2; i>0; i--)
        for(j=0; j<layer_size[i]; j++)
            for(k=0; k<layer_size[i+1]; k++)
                error[i][j] += error[i+1][k]*weight[i][j][k];
}// back pass



void Net::backpropagation(double learning_rate, int num_data){
    int i, j, k, cycle;
    
    for(cycle=1; cycle<num_data; cycle++)
        for(i=0; i<num_layer-1; i++)
            for(j=0; j<layer_size[i]; j++)
                error[0][i+1][j] = (error[0][i+1][j] + error[cycle][i+1][j])/num_data;
    
    
    
    // update weight
#pragma omp parallel for num_threads(nCPU)
    for(i=0; i<num_layer-1; i++)
        for(j=0; j<layer_size[i]; j++)
            for(k=0; k<layer_size[i+1]; k++)
                weight[i][j][k] -= error[0][i+1][k]				// delta
                * value[num_data-1][i+1][k]*(1-value[num_data-1][i+1][k])	// d{simoid(e)}/ d{e}
                * value[num_data-1][i][j]					// x1
                * learning_rate;						// learning rate
    
    
    // update bias
#pragma omp parallel for num_threads(nCPU)
    for(i=1; i<num_layer-1; i++)
        for(j=0; j<layer_size[i]; j++)
            for(k=0; k<layer_size[i+1]; k++)
                bias[i][j] += error[0][i+1][k]				// delta
                * value[num_data-1][i+1][k]*(1-value[num_data-1][i+1][k])	// d{simoid(e)}/ d{e}
                * value[num_data-1][i][j]					// x1
                * learning_rate;						// learning rate
}// back propagation



// Destructor
Net::~Net(){
    int i, j;
    
    // destroy weight
    for(int i=0; i<num_layer-1; i++)
        for(int j=0; j<layer_size[i]; j++)
            delete[] weight[i][j];
    for(int i=0; i<num_layer-1; i++)
        delete[] weight[i];
    delete[] weight;
    
    // destroy bias
    for(int i=1; i<num_layer; i++)
        delete[] bias[i];
    delete[] bias;
    
    // delete error
    for(i=0; i<mini_batch_size; i++)
        for(j=1; j<num_layer; j++)
            delete[] error[i][j];
    for(int i=0; i<mini_batch_size; i++)
        delete[] error[i];
    delete[] error;
    
    // destroy value
    for(i=0; i<mini_batch_size; i++)
        for(j=0; j<num_layer; j++)
            delete[] value[i][j];
    for(i=0; i<mini_batch_size; i++)
        delete[] value[i];
    delete[] value;
    
    // destroy layer size
    delete[] layer_size;
}// destructor

#endif /* learner_h */