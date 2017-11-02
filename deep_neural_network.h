/*
 * Deep Neural Network definition file
 *
 * This deep neural network will do un-supervised training.
 * It will use stochastic gradient descent (not full gradient descent)
 *
 */

#ifndef DEEP_NEURAL_NETWORK
#define DEEP_NEURAL_NETWORK

#include <windows.h>			/* for multi threaded stuff */

enum nn_type_enum {SIGMOID, RELU};

class deep_neural_network
{
	/*
	 * A deep neural network is a series of vectors plus a "squashing" function
	 */

public:
	deep_neural_network(int n_in, int n_out, double learning_rate, double momentum, int n_hidden_layers, int * hidden_layers_sizes, int thread_count, nn_type_enum nn_type);
	~deep_neural_network();
	
	void init();
	
	void dispose();
	
	
	void pre_train(double * input, int layer_to_pre_train);
	
	void backpropagation(double * input, double * target);
	
	void compute_output(double * input, double * output);
	void compute_output_pre_train(double * input, int layer_to_calc, double * output);
	double compute_pre_train_error(double * input, int layer_to_test);		/* a helper function used for debuging */
	
	int hidden_layer_size(int hidden_layer_count) {return this->hidden_layers_sizes[hidden_layer_count];};
	
	void save_to_file(char * file_name);
	deep_neural_network * read_from_file(char * file_name);
	
	bool identical_neural_networks(deep_neural_network * cmp);
	
	void fake_back_propagation();								/* a helper function to test some functionality */ 
	void print_neural_network();

private:
	int n_in;													/* size of the input vector */
	int n_out;													/* size of the output vector */
	double learning_rate;										/* the learning rate */
	double momentum;											/* momentum. For now we do not assume exponential decay */
	int thread_count;											/* vector multiplication is very easy to run in parallel. so we can use more threads */
	nn_type_enum nn_type;										/* the type of the neural network "squashing" function. Currently RELU or Sigmoid */
	int n_hidden_layers;										/* how deep the neural network is(how many hidden layers does the neural network have) */
	int * hidden_layers_sizes;									/* each integer will be the number of hidden nodes in a layer in the network. */

	double ** hidden_layers_weights;							/* the neural network. The matrices corresponding to different layers */
	double ** prior_descent_hidden_layers_deltas;				/* the derivatives from the prior descent. Used during pre_training / backpropagation */
	
	/*
	 * Arrays we will use over and over during pre-training. Pre-training takes 10 of thousands of inputs and each input
	 * needs pre-training.
	 * We re-use these arrays so we allocate as rarely as possible.
	 */
	double * pt_intermediary_inp;								/* for when we pre-train deeper than 1 layer */
	double * pt_hidden_units;									/* an intermediary layer in pre-training */
	double * pt_output_units;									/* compute the output with the intermediary state */
	double * pt_output_derivatives;								/* upper layer derivatives */
	double * pt_hidden_derivatives;								/* lower layer derivatives */
	double * pt_upper_weights;									/* intermediary weights. the size is equal to size of prior layer x size of current layer*/
	double * pt_upper_weights_prior_descent_deltas;				/* intermediary deltas from prior descents */
	int curr_pre_train_layer;									/* keep track of current layer to pre-train. upper_weights need to be re-set when pre-training a different layer */
	
	
	/*
	 * Arrays to re-use over and over during backpropagation.
	 * Each input requires a few backpropagations. And there are 10s of thousands of inputs.
	 * We re-use these arrays as much as we can so we allocate as rarely as possible.
	 */
	double ** bp_forward_feed_units;							/* the intermediary steps (units) from calculating the output */
	double * bp_upper_derivatives;								/* the upper layer derivatives to be used in the chain rule */
	double * bp_curr_derivatives;								/* current layer derivatives to be used in the chain rule */
	
	/*
	 * Arrays to re-use over and over during compute_output.
	 * We have to compute 10s of thousands of inputs.
	 * We re-use these arrays as much as possible so we allocate as rarely as possible
	 */
	double * compute_output_res;								/* results for current level computation */
	double * compute_output_prior_res;							/* results for  prior level computation. these are inputs into the current level computation. */
	int compute_output_res_sz;									/* size of the current level computation array */
	int compute_output_prior_res_sz;							/* size of the prior level computation array */

	double squashing_function(double input);					/* the "squashing" function */
	double squashing_function_derivative(double func_value);	/* the derivative of the "squashing" function. The derivative for some squashing functions are */
																/* easy to calculate from the value of squashing function */
	
	int adjust_for_squashing;									/* the "squashing" function converges very quickly to its limit. */
																/* We have to adjust (divide by) the argument passed to the "squashing" function */
	
	/* some helper functions */
	double vector_multiplication(double * in_1, double * in_2, int len);	/* our vector multiplication function */
	double * grow_array(double * src, int src_sz, int new_sz);	/* a function to grow the current array if necessary */
	void set_memory_for_pre_train();							/* a function to set all the memory ready for pre-training */
	void reset_pre_trained_state(int layer_to_pre_train);		/* inner weights correspond to a certain pre-trained layer. reset them when training on a different layer */
	void set_memory_for_backpropagation();						/* a function to set all the memory required for backpropagation */
	
	
	/* multi-threaded stuff */
	static DWORD WINAPI worker_thread(void * in);				/* the worker thread */
	void dispose_threads();										/* dispose the threads when we are taking down the nn */
	struct thread_info
	{
		double * v1;											/* first vector to multiply */
		double * v2;											/* second vector to multiply */
		int v_sz;												/* size of the vectors to multiply */
		double res;												/* result of multiplication */
		HANDLE wait_for_work;									/* mutex to wait for work. Signalled by main thread. */
		HANDLE work_done;										/* mutex to signal work is done.  Signalled by worker thread. */
		bool exit_thread;										/* main thread sets to TRUE when thread is done. Done nn is being disposed */
	};
	thread_info * thread_infos;									/* info passed to threads */
	HANDLE * threads;											/* thread HANDLES. from CreateThread function */
}; // deep_neural_network

#endif /* DEEP_NEURAL_NETWORK */