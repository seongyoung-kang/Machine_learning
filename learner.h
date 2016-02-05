#ifndef learner_h
#define learner_h

#define INPUT_SIZE 4
#define OUTPUT_SIZE 16

#define LEARNING_RATE 0.5

#define ISTEST 0

/******************************************************
 *
 * num_layer : number of layers (include input, output layer)
 * layer_size : array for each layers size
 * value :
 *
 *
 * mini_batch : deltas for each cycles
 *
 *
 ******************************************************/

class Net{
private:
    int epoch;
    int mini_batch_size;
    int num_layer;
    int* layer_size;
    double** value;
    double*** weight;
    double** bias;
    double** error;
    double*** mini_batch;
    
public:
    Net(int* layer_size, int num_layer, int mini_batch_size, int epoch);
    ~Net();
    
    void train(int data_size, double input[][INPUT_SIZE], double desired[][OUTPUT_SIZE]);
    double* test(double* input);
    
private:
    void initializer();
    
    double sigmoid(double num);
    
    void feedforward(double* input);
    void back_pass(double* desired);
    void backpropagation(double learning_rate);
    
};// Net class

















//
//
//		implementations

// constructor
Net::Net(int* layer_size, int num_layer, int mini_batch_size, int epoch){
    //
    // write on class vars
    this->num_layer = num_layer;
    this->layer_size = layer_size;
    this->mini_batch_size = mini_batch_size;
    this->epoch = epoch;
    
    
    //
    //          memory allocation
    // value
    value = new double*[num_layer];
    for(int i=0; i<num_layer; i++)
        value[i] = new double[layer_size[i]];
    
    // error
    error = new double*[num_layer];
    for(int i=1; i<num_layer; i++)
        error[i] = new double[layer_size[i]];
    
    
    // bias
    bias = new double*[num_layer];
    for(int i=1; i<num_layer; i++)
        bias[i] = new double[layer_size[i]];  //  bias[0] : trash
    
    
    // weight layer
    weight = new double**[num_layer-1];
    
    // weight from
    for(int i=0; i<num_layer-1; i++)
        weight[i] = new double*[layer_size[i]];
    
    // weight to
    for(int i=0; i<num_layer-1; i++)
        for(int j=0; j<layer_size[i]; j++)
            weight[i][j] = new double[layer_size[i+1]];
    
    
    // mini batch
    mini_batch = new double**[mini_batch_size];
    for(int i=0; i<mini_batch_size; i++)
        mini_batch[i] = new double*[num_layer];
    for(int i=0; i<mini_batch_size; i++)
        for(int j=0; j<num_layer; j++)
            mini_batch[i][j] = new double[layer_size[j]];
    
    
    //
    //
    //          input random initial weights&biases(-1 ~ 1)
    for(int i=0; i<num_layer-1; i++)
        for(int j=0; j<layer_size[i]; j++)
            for(int k=0; k<layer_size[i+1]; k++)
                weight[i][j][k] = (double)rand()/(RAND_MAX/2)-1;
    
    for(int i=1; i<num_layer; i++)
        for(int j=0; j<layer_size[i]; j++)
            bias[i][j] = (double)rand()/(RAND_MAX/2)-1;
    
}// constructor



void Net::initializer(){
    for(int i=0; i<num_layer; i++)
        for(int j=0; j<layer_size[i]; j++)
            value[i][j] = 0;
    
    for(int i=1; i<num_layer; i++)
        for(int j=0; j<layer_size[i]; j++)
            error[i][j] = 0;
}// initializer



double Net::sigmoid(double num){
    return 1/(1 + exp(-num));
}// sigmoid



void Net::feedforward(double* input){
    for(int i=0; i<layer_size[0]; i++)
        value[0][i] = input[i];
    
    double sum;
    for(int i=0; i<num_layer-1; i++)
        for(int k=0; k<layer_size[i+1]; k++){
            sum = 0;
            for(int j=0; j<layer_size[i]; j++)
                sum += weight[i][j][k]*value[i][j];
            value[i+1][k] = sigmoid(sum + bias[i+1][k]);
        }
}// feed forward



void Net::back_pass(double* desired){
    for(int i=0; i<layer_size[num_layer-1]; i++)
        error[num_layer-1][i] = value[num_layer-1][i] - desired[i];
    
    for(int i=num_layer-2; i>0; i--)
        for(int j=0; j<layer_size[i]; j++)
            for(int k=0; k<layer_size[i+1]; k++)
                error[i][j] += error[i+1][k]*weight[i][j][k];
    
}// back pass



void Net::backpropagation(double learning_rate){
    for(int cycle=0; cycle<mini_batch_size; cycle++){
        
        // update weight
        for(int i=0; i<num_layer-1; i++)
            for(int j=0; j<layer_size[i]; j++)
                for(int k=0; k<layer_size[i+1]; k++)
                    weight[i][j][k] -= mini_batch[cycle][i+1][k]				// delta
                    * value[i+1][k]*(1-value[i+1][k])		// d{simoid(e)}/ d{e}
                    * value[i][j]							// x1
                    * learning_rate;						// learning rate
        // update bias
        for(int i=1; i<num_layer-1; i++)
            for(int j=0; j<layer_size[i]; j++)
                for(int k=0; k<layer_size[i+1]; k++)
                    bias[i][j] += mini_batch[cycle][i+1][k]				// delta
                    * value[i+1][k]*(1-value[i+1][k])	// d{simoid(e)}/ d{e}
                    * value[i][j]						// x1
                    * learning_rate;					// learning rate
    }
}// back propagation


void Net::train(int data_size, double input[][INPUT_SIZE], double desired[][OUTPUT_SIZE]){
    for(int i=0; i<epoch; i++){
        for(int j=0; j<data_size;){
            for(int cycle=0; cycle<mini_batch_size && j<data_size; cycle++, j++){
                initializer();
                feedforward(input[j]);
                back_pass(desired[j]);
                
                for(int k=1; k<num_layer; k++)			// copy from error to mini_batch
                    for(int l=0; l<layer_size[k]; l++)
                        mini_batch[cycle][k][l] = error[k][l];
            }
            backpropagation(LEARNING_RATE);
        }
    }
}// train


double* Net::test(double* input){
    initializer();
    feedforward(input);
    return value[num_layer-1];
}// test




// Destructor
Net::~Net(){
    // destry weight
    for(int i=0; i<num_layer-1; i++)
        for(int j=0; j<layer_size[i]; i++)
            delete[] weight[i][j];
    for(int i=0; i<num_layer; i++)
        delete[] weight[i];
    delete[] weight;
    
    // destroy bias
    for(int i=0; i<num_layer; i++)
        delete[] bias[i];
    delete[] bias;
    
    // delete error
    for(int i=0; i<num_layer; i++)
        delete error[i];
    delete[] error;
    
    // destroy value
    for(int i=0; i<num_layer; i++)
        delete[] value[i];
    delete[] value;
    
    // destroy mini_batch
    for(int i=0; i<mini_batch_size; i++)
        for(int j=0; j<num_layer; j++)
            delete[] mini_batch[i][j];
    for(int i=0; i<num_layer; i++)
        delete[] mini_batch;
    delete[] mini_batch;
    
    // destroy layer_size
    delete[] layer_size;
}// destructor



#endif /* learner_h */