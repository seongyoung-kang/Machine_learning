#ifndef learner_h
#define learner_h

#define NUM_LAYER 4

#define INPUT_SIZE 4
#define HIDDEN_SIZE 20, 20
#define OUTPUT_SIZE 16

#define LEARNING_RATE 0.5


using namespace std;

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
    int num_layer;
    int mini_batch_size;
    int* layer_size;
    double** value;
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
    
    void feedforward(double* input);
    void back_pass(double* desired, double** error);
    void backpropagation(double learning_rate, int num_data);
    
};// Net class







//
//
//		implementations

// constructor
Net::Net(int* layer_size, int num_layer, int mini_batch_size, int epoch){
    //
    // write on class vars
    this->num_layer = num_layer;
    this->mini_batch_size = mini_batch_size;
    
    
    //
    //          memory allocation
    // layer size
    this->layer_size = new int[num_layer];
    for(int i=0; i<num_layer; i++)
        this->layer_size[i] = layer_size[i];
    // value
    value = new double*[num_layer];
    for(int i=0; i<num_layer; i++)
        value[i] = new double[layer_size[i]];
    
    // error
    error = new double**[mini_batch_size];
    for(int i=0; i<mini_batch_size; i++)
        error[i] = new double*[num_layer];
    for(int i=0; i<mini_batch_size; i++)
        for(int j=1; j<num_layer; j++)
            error[i][j] = new double[layer_size[j]];
    
    
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
    
    //
    //
    //          input random initial weights & biases(-1 ~ 1)
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
    
    for(int k=0; k<mini_batch_size; k++)
        for(int i=1; i<num_layer; i++)
            for(int j=0; j<layer_size[i]; j++)
                error[k][i][j] = 0;
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



void Net::back_pass(double* desired, double** error){
    for(int i=0; i<layer_size[num_layer-1]; i++)
        error[num_layer-1][i] = value[num_layer-1][i] - desired[i];
    
    for(int i=num_layer-2; i>0; i--)
        for(int j=0; j<layer_size[i]; j++)
            for(int k=0; k<layer_size[i+1]; k++)
                error[i][j] += error[i+1][k]*weight[i][j][k];
    
}// back pass



void Net::backpropagation(double learning_rate, int num_data){
    for(int cycle=0; cycle<num_data; cycle++){
        // update weight
        for(int i=0; i<num_layer-1; i++)
            for(int j=0; j<layer_size[i]; j++)
                for(int k=0; k<layer_size[i+1]; k++)
                    weight[i][j][k] -= error[cycle][i+1][k]			// delta
                    * value[i+1][k]*(1-value[i+1][k])				// d{simoid(e)}/ d{e}
                    * value[i][j]									// x1
                    * learning_rate;								// learning rate
        
        // update bias
        for(int i=1; i<num_layer-1; i++)
            for(int j=0; j<layer_size[i]; j++)
                for(int k=0; k<layer_size[i+1]; k++)
                    bias[i][j] += error[cycle][i+1][k]			// delta
                    * value[i+1][k]*(1-value[i+1][k])			// d{simoid(e)}/ d{e}
                    * value[i][j]								// x1
                    * learning_rate;							// learning rate
    }
}// back propagation


void Net::train(double input[][INPUT_SIZE], double desired[][OUTPUT_SIZE], int num_data){
    initializer();
    for(int i=0; i<num_data; i++){
        feedforward(input[i]);
        back_pass(desired[i], error[i]);
        
        for(int j=0; j<4; j++)
	        cout << input[i][j];
        cout << endl;
    }
    backpropagation(LEARNING_RATE, num_data);
}// train


double* Net::test(double* input){
    initializer();
    feedforward(input);
    return value[num_layer-1];
}// test



// Destructor
Net::~Net(){
    // destroy weight
    for(int i=0; i<num_layer-1; i++)
        for(int j=0; j<layer_size[i]; j++)
            delete[] weight[i][j];
    for(int i=0; i<num_layer-1; i++)
        delete[] weight[i];
    delete[] weight;
    
    // destroy bias
    for(int i=0; i<num_layer; i++)
        delete[] bias[i];
    delete[] bias;
    
    // delete error
    for(int i=0; i<mini_batch_size; i++)
        for(int j=0; j<num_layer; j++)
            delete[] error[i][j];
    for(int i=0; i<mini_batch_size; i++)
        delete[] error[i];
    delete[] error;
    
    // destroy value
    for(int i=0; i<num_layer; i++)
        delete[] value[i];
    delete[] value;
    
    // destroy layer size
    delete[] layer_size;
}// destructor



#endif /* learner_h */
