/*
 *
 * Implementation file for deep neural network
 *
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "deep_neural_network.h"

deep_neural_network::deep_neural_network(int n_in, int n_out, double learning_rate, double momentum, int n_hidden_layers, int * hidden_layers_sizes, int thread_count, nn_type_enum nn_type)
{
	/* tabula rasa */
	memset (this /*dest*/, 0 /*value*/, sizeof(deep_neural_network) /*num*/);

	this->n_in = n_in;
	this->n_out = n_out;
	this->learning_rate = learning_rate;
	this->momentum = momentum;
	this->nn_type = nn_type;
	
	this->n_hidden_layers = n_hidden_layers;
	
	this->hidden_layers_sizes = (int *) malloc (sizeof(int) * n_hidden_layers);
	
	int layer_sizes_sum = 0;   /* we will need this for the adjust parameter */
	
	for (int i = 0; i < n_hidden_layers; i ++)
	{
		layer_sizes_sum += hidden_layers_sizes[i];
		this->hidden_layers_sizes[i] = hidden_layers_sizes[i];
	}
	
	/* no one should need more than 32 threads */
	if (thread_count > 32)
	{
		this->thread_count = 32;
	}
	else
	{
		this->thread_count = thread_count;
	}
	
	/* we assume we are using the sigmoid function. Sigmoid becomes 1 when its input is around 30 */
	/* the parameter passed to the sigmoid is usually less than the number of nodes */
	/* so the adjusting factor is the number of nodes divided by 30 */
	this->adjust_for_squashing = (layer_sizes_sum / 30) + 1;
}

deep_neural_network::~deep_neural_network()
{
	/* nothing to do */
	/* all the work is done in the dispose method */
}

void deep_neural_network::init()
{
	/* 
	 * init allocates the space for weights, 
	 * initializes them to small random numbers,
	 * allocates space for prior descent derivatives,
	 * and initializes prior descent derivatives to 0.
	 */


	/*
	 * The first hidden nodes have dimension (n_in + 1) x first hidden layer size.
	 * The second hidden nodes have dimension first hidden layer size x second hidden layer size
	 * .....
	 * Last hidden nodes have dimentions last hidden layer size x (n_out)
	 *
	 * So the total number of hidden weights are n_hidden_layers + 1
	 */
    this->hidden_layers_weights = (double **) malloc((this->n_hidden_layers+1) * sizeof(double *));
	this->prior_descent_hidden_layers_deltas = (double **) malloc((this->n_hidden_layers+1) * sizeof(double *));
	
	/* 
	 * The first layer has the bias weight.
	 * The first dimension of the first layer has the same size as the input.
	 * Second dimension of the first layer has the same size as first layer.
	 */
	int size = (this->n_in + 1) * (this->hidden_layers_sizes[0]);
	this->hidden_layers_weights[0] = (double *) malloc(size * sizeof(double));
	srand(time(NULL));
	double * current_layer = this->hidden_layers_weights[0];
	for (int i = 0; i < size; i++)
	{
		/*initialize initial weights to small (less than 0.01) values.*/
		current_layer[i] = ((double) rand()) / (RAND_MAX + 1) / 100;
	}
	
	this->prior_descent_hidden_layers_deltas[0] = (double *) malloc(size * sizeof(double));
	memset(this->prior_descent_hidden_layers_deltas[0] /*dest*/, 0 /*value*/, size * sizeof(double));
	
	/*n_hidden_layers-1 because the first layer has the same size as the input, second layer has the same size as the first layer; third layer has the same size as etc....*/
	for (int i = 1; i < this->n_hidden_layers; i++)		
	{
		size = (this->hidden_layers_sizes[i-1]) * (this->hidden_layers_sizes[i]);
		current_layer = (double *) malloc(size * sizeof(double));
		this->hidden_layers_weights[i] = current_layer;
		for (int j = 0; j < size; j++)
		{
			/*initialize initial weights to small (less than 0.01) values.*/
			current_layer[j] = ((double) rand()) / (RAND_MAX + 1) / 100;
		}

		this->prior_descent_hidden_layers_deltas[i] = (double *) malloc(size * sizeof(double));
		memset(this->prior_descent_hidden_layers_deltas[i] /*dest*/, 0 /*value*/, size * sizeof(double));
	}
	
	/*output of the last layer is n_out*/
	size = (this->hidden_layers_sizes[this->n_hidden_layers-1]) * this->n_out;
	current_layer = (double *) malloc(size * sizeof(double));
	this->hidden_layers_weights[this->n_hidden_layers] = current_layer;
	for (int j = 0; j < size; j++)
	{
		/*initialize initial weights to small (less than 0.01) values.*/
		current_layer[j] = ((double) rand()) / (RAND_MAX + 1) / 100;
	}
	this->prior_descent_hidden_layers_deltas[this->n_hidden_layers] = (double *) malloc(size * sizeof(double));
	memset(this->prior_descent_hidden_layers_deltas[this->n_hidden_layers] /*dest*/, 0 /*value*/, size * sizeof(double));
	
	/* multi threaded stuff... */
	/* if the thread count is greater than 0 we will create an array of threads. Otherwise, the nn is single-threaded*/
	if (this->thread_count > 1)
	{
		this->threads = (HANDLE *) malloc (sizeof(HANDLE) * this->thread_count);
		this->thread_infos = (thread_info *) malloc (sizeof(thread_info) * this->thread_count);
		for (int i = 0; i < this->thread_count; i++)
		{
			thread_info * ti = this->thread_infos + i;
			memset(ti /*dest*/, 0 /*value*/, sizeof(thread_info) /*num*/);
			ti->wait_for_work = CreateSemaphore(NULL /*default security attributes*/,
											   0 /*initially un-signaled*/, 
											   1 /*max sem count*/, 
											   NULL /*name*/);
			ti->work_done = CreateSemaphore(NULL /*default security attributes*/,
										   0 /*initially un-signaled*/, 
										   1 /*max sem count*/, 
										   NULL /*name*/);
			this->threads[i] = CreateThread(NULL, 								/*default security attributes*/
											0, 									/*default stack size*/
											deep_neural_network::worker_thread, /*thread function*/
											(void *) (ti),	 					/*thread function arguments*/
											0, 									/*default creation flags*/
											NULL); 								/*we don't need the thread id */
		}
	}
}

void deep_neural_network::dispose()
{
	if (this->hidden_layers_sizes)
	{
		free(this->hidden_layers_sizes);
	}
	
	if (this->hidden_layers_weights)
	{
		for (int i = 0; i < this->n_hidden_layers; i++)
		{
			if (this->hidden_layers_weights[i])
			{
				free(this->hidden_layers_weights[i]);
			}
		}
		free(this->hidden_layers_weights);
	}
	
	if (this->prior_descent_hidden_layers_deltas)
	{
		for (int i = 0; i < this->n_hidden_layers; i++)
		{
			if (this->prior_descent_hidden_layers_deltas[i])
			{
				free(this->prior_descent_hidden_layers_deltas[i]);
			}
		}
		free(this->prior_descent_hidden_layers_deltas);
	}
	
	/* free all the pre train buffers */
	if (this->pt_intermediary_inp)
	{
		free(this->pt_intermediary_inp);
	}
	if (this->pt_hidden_units)
	{
		free(this->pt_hidden_units);
	}
	if (this->pt_output_units)
	{
		free(this->pt_output_units);
	}
	if (this->pt_output_derivatives)
	{
		free(this->pt_output_derivatives);
	}
	if (this->pt_hidden_derivatives)
	{
		free(this->pt_hidden_derivatives);
	}
	if (this->pt_upper_weights)
	{
		free(this->pt_upper_weights);
	}
	if (this->pt_upper_weights_prior_descent_deltas)
	{
		free(pt_upper_weights_prior_descent_deltas);
	}

	/* free all the backpropagation buffers */
	if (this->bp_forward_feed_units)
	{
		/* bp_forward_feed_units has the same layout like the hidden weigths array */
		int i = this->n_hidden_layers + 1;		/* +1 because the last layer is the output layer. */
		for (int j = 0; j < i; j++)
		{
			if (this->bp_forward_feed_units[j])
			{
				free(this->bp_forward_feed_units[j]);
			}
		}
		free(this->bp_forward_feed_units);
	}
	
	if (this->bp_upper_derivatives)
	{
		free (this->bp_upper_derivatives);
	}

	if (this->bp_curr_derivatives)
	{
		free (this->bp_curr_derivatives);
	}
	
	/* free the compute output arrays */
	if (this->compute_output_res)
	{
		free (this->compute_output_res);
	}
	if (this->compute_output_prior_res)
	{
		free (this->compute_output_prior_res);
	}
	
	/* dispose the threads */
	this->dispose_threads();

	/* tabula rasa */
	memset (this /*dest*/, 0 /*value*/, sizeof(deep_neural_network) /*num*/);
}

void deep_neural_network::set_memory_for_pre_train()
{
	/* we allocate the memory only once per lifetime */
	if (this->pt_intermediary_inp)
	{
		return;
	}
	
	int max_layer_sz = this->n_in + 1;
	int max_weight_sz = max_layer_sz * this->hidden_layers_sizes[0];
	
	for (int i = 0; i < this->n_hidden_layers; i ++)
	{
		int curr_layer_sz = this->hidden_layers_sizes[i];
		if (curr_layer_sz > max_layer_sz)
		{
			max_layer_sz = curr_layer_sz;
		}
		if (i > 0)
		{
			int curr_weight_sz = curr_layer_sz * this->hidden_layers_sizes[i-1];
			if (curr_weight_sz > max_weight_sz)
			{
				max_weight_sz = curr_weight_sz;
			}
		}
	}
	
	this->pt_intermediary_inp = (double *) malloc (max_layer_sz * sizeof(double));
	this->pt_hidden_units = (double *) malloc (max_layer_sz * sizeof(double));
	this->pt_output_units = (double *) malloc (max_layer_sz * sizeof(double));
	this->pt_output_derivatives = (double *) malloc (max_layer_sz * sizeof(double));
	this->pt_hidden_derivatives = (double *) malloc (max_layer_sz * sizeof(double));
	this->pt_upper_weights = (double *) malloc (max_weight_sz * sizeof(double));
	this->pt_upper_weights_prior_descent_deltas = (double *) malloc (max_weight_sz * sizeof(double));
}

void deep_neural_network::reset_pre_trained_state(int layer_to_pre_train)
{
	int sz = (layer_to_pre_train == 1) ? this->n_in + 1 : this->hidden_layers_sizes[layer_to_pre_train-2];
	sz = sz * this->hidden_layers_sizes[layer_to_pre_train-1];
	double * curr_res = this->pt_upper_weights;
	
	for (int i = 0; i < sz; i++)
	{
		curr_res[i] = ((double) rand()) / (RAND_MAX + 1) / 100;
	
	}
	
	memset(this->pt_upper_weights_prior_descent_deltas /*dest*/, 0 /*value*/,sz * sizeof(double) /*num*/);
}

/* This is unsupervised pre-training; the expected output is the same as the input */
/* input in an array with n_in + 1 values, where the first value is 1. This way vector multiplication is easier to code.*/
void deep_neural_network::pre_train(double * input, int layer_to_pre_train)
{
	/*
	 * The pre-training algorithm:
	 * 1. Calculate the hidden units and output units
	 * 2. Calculate the output and hidden derivatives
	 * 3. Take small steps opposite the derivatives
	 *
	 * Most of the complexity of this function is memory book-keeping so that we can re-use as much memory as possible.
	 */
	
	/* TODO: some of the loops below can be "squashed" together. Think about that */
	
	if (abs(input[0] - 1) > 0.0000001)
	{
		printf ("the first value in the input should be 1. Pre training failed.\n");
		return;
	}
	
	this->set_memory_for_pre_train();
	if (layer_to_pre_train != this->curr_pre_train_layer)
	{
		this->reset_pre_trained_state(layer_to_pre_train);
	}
	this->curr_pre_train_layer = layer_to_pre_train;
	
	/* cache some pointers */
	int * ls = this->hidden_layers_sizes;
	double ** hw = this->hidden_layers_weights;

	/* size of what is considered input into this stage of pre-training */
	int inp_sz = (layer_to_pre_train == 1) ? this->n_in + 1 : ls[layer_to_pre_train - 2];	/* input into unsupervised learning is the size of prior layer */
	double * pt_inp, * curr_weights, * curr_res, * curr_inp;

	/* size of the layer being pre trained */
	int pt_sz = ls[layer_to_pre_train - 1];
	
	/* 1. Calculate the hidden units and output units */
	/* 1. Calculate the hidden units */
	if (layer_to_pre_train == 1)
	{
		this->compute_output_pre_train(input, layer_to_pre_train, this->pt_hidden_units);
		pt_inp = input;
	}
	else
	{
		/* intermediary layers pre-training have intermediary inputs */
		this->compute_output_pre_train(input, layer_to_pre_train - 1, this->pt_intermediary_inp);
		pt_inp = this->pt_intermediary_inp;
		curr_res = this->pt_hidden_units;
		curr_weights = hw[layer_to_pre_train-1];
		
		for (int i = 0; i < pt_sz; i ++)
		{
			double tmp = this->vector_multiplication(pt_inp, curr_weights + i * inp_sz, inp_sz);
			tmp = tmp / this->adjust_for_squashing;
			tmp = this->squashing_function(tmp);
			curr_res[i] = tmp;
		}
	}
	
	/* 1. Calculate the output units */
	/* cache some pointers */
	curr_inp = this->pt_hidden_units;
	curr_res = this->pt_output_units;
	curr_weights = this->pt_upper_weights;
	for (int i = 0; i < inp_sz; i++)
	{
		double tmp = this->vector_multiplication(curr_inp, curr_weights + i * pt_sz, pt_sz);
		tmp = tmp / this->adjust_for_squashing;
		tmp = this->squashing_function(tmp);
		curr_res[i] = tmp;
	}

	
	/* 2. Calculate the output derivatives and hidden derivatives */
	/* 2. Calculate the output derivatives */
	/* cache some pointers */
	curr_res = this->pt_output_derivatives;
	curr_inp = this->pt_output_units;
	for (int i = 0; i < inp_sz; i++)
	{
		double tmp = (pt_inp[i] - curr_inp[i]);		/* unsupervised learning: target is the same as input */
		tmp = this->squashing_function_derivative(curr_inp[i]) * tmp;
		tmp = tmp / this->adjust_for_squashing;
		curr_res[i] = tmp;
	}
	
	/* 2. Calculate the hidden derivatives */
	/* cache some pointers */
	curr_res = this->pt_hidden_derivatives;
	curr_inp = this->pt_output_derivatives;
	double * hidden_units = this->pt_hidden_units;
	curr_weights = this->pt_upper_weights;
	for (int i = 0; i < pt_sz; i ++)
	{
		double tmp = this->vector_multiplication(curr_inp, curr_weights + i * inp_sz, inp_sz);
		tmp = tmp * this->squashing_function_derivative(hidden_units[i]);
		tmp = tmp / this->adjust_for_squashing;
		curr_res[i] = tmp;
	}
	
	/* 3. Take small steps opposite the derivatives */
	/* 3. Take small steps in the output layer */
	/* cache some pointers */
	curr_res = this->pt_upper_weights;
	curr_inp = this->pt_hidden_units;
	double * prior_deltas = this->pt_upper_weights_prior_descent_deltas;
	double * curr_derivatives = this->pt_output_derivatives;

	double * curr_res2 = hw[layer_to_pre_train - 1];
	double * curr_inp2 = pt_inp;
	double * curr_derivatives2 = this->pt_hidden_derivatives;
	double * prior_deltas2 = this->prior_descent_hidden_layers_deltas[layer_to_pre_train - 1];

	double lr = this->learning_rate;
	double moment = this->momentum;

	for (int i = 0; i < pt_sz; i++)
	{
		for (int j = 0; j < inp_sz; j++)
		{
			int index1 = i + j * pt_sz;
			/* take small step in the output layer */
			double delta = lr * curr_inp[i] * curr_derivatives[j];
			delta += moment * prior_deltas[index1];
			curr_res[index1] += delta;
			prior_deltas[index1] = delta;
			
			/* take small step in the inner layer */
			int index2 = j + i * inp_sz;
			double delta2 = lr * curr_inp2[j] * curr_derivatives2[i];
			delta2 += moment * prior_deltas2[index2];
			curr_res2[index2] += delta2;
			prior_deltas2[index2] = delta2;
		}
	}
}

double deep_neural_network::compute_pre_train_error(double * input, int pre_train_layer_to_test)
{
	/* this function will be very similar to the first part of the pre_train function */
	/* just like the pre_train, this function will calculate the intermediary results */
	/* after that it will not go into calculating the derivatives - it will just calculate the */
	/* error and output it */
	
	/* cache some pointer */
	double ** hw = this->hidden_layers_weights;
	int * ls = this->hidden_layers_sizes;
	
	int inp_sz = (pre_train_layer_to_test == 1) ? this->n_in + 1 : ls[pre_train_layer_to_test - 2];	/* input into unsupervised learning is the size of prior layer */
	int pt_sz = ls[pre_train_layer_to_test - 1];

	double * pt_inp, * curr_weights, * curr_res, * curr_inp;

	double res = 0.0;

	/* 1. Calculate the hidden units */
	if (pre_train_layer_to_test == 1)
	{
		this->compute_output_pre_train(input, pre_train_layer_to_test, this->pt_hidden_units);
		pt_inp = input;
	}
	else
	{
		/* intermediary layers pre-training have intermediary inputs */
		this->compute_output_pre_train(input, pre_train_layer_to_test - 1, this->pt_intermediary_inp);
		pt_inp = this->pt_intermediary_inp;
		curr_res = this->pt_hidden_units;
		curr_weights = hw[pre_train_layer_to_test-1];
		
		for (int i = 0; i < pt_sz; i ++)
		{
			double tmp = this->vector_multiplication(pt_inp, curr_weights + i * inp_sz, inp_sz);
			tmp = tmp / this->adjust_for_squashing;
			tmp = this->squashing_function(tmp);
			curr_res[i] = tmp;
		}
	}
	
	/* 1. Calculate the output units */
	/* cache some pointers */
	curr_inp = this->pt_hidden_units;
	curr_res = this->pt_output_units;
	curr_weights = this->pt_upper_weights;
	for (int i = 0; i < inp_sz; i++)
	{
		double tmp = this->vector_multiplication(curr_inp, curr_weights + i * pt_sz, pt_sz);
		tmp = tmp / this->adjust_for_squashing;
		tmp = this->squashing_function(tmp);
		curr_res[i] = tmp;
	}

	
	/* 2. Calculate the error */
	/* cache some pointers */
	curr_res = this->pt_output_derivatives;
	curr_inp = this->pt_output_units;
	for (int i = 0; i < inp_sz; i++)
	{
		res += (pt_inp[i] - curr_inp[i]) * (pt_inp[i] - curr_inp[i]);
	}
	
	return res;
}

void deep_neural_network::set_memory_for_backpropagation()
{
	/* memory for back propagation needs to be setup once per neural network lifetime */
	if (this->bp_forward_feed_units)
	{
		return;
	}

	int layer_count = this->n_hidden_layers;
	int * sizes_arr = this->hidden_layers_sizes;
	int max_hidden_layer_sz = 0;

	/* the forward feed units have are an array of arrays of double. The inner arrays have the same size as the hidden layers sizes */
	this->bp_forward_feed_units = (double **) malloc ((layer_count+1) * sizeof (double *));
	for (int i = 0; i < layer_count; i ++)
	{
		this->bp_forward_feed_units[i] = (double *) malloc (sizes_arr[i] * sizeof(double));
		if (sizes_arr[i] > max_hidden_layer_sz)
		{
			max_hidden_layer_sz = sizes_arr[i];
		}
	}
	
	if (this->n_out > max_hidden_layer_sz)
	{
		max_hidden_layer_sz = this->n_out;
	}
	
	/* last layer in the forward feed units will be the results of the neural network. */
	/* will have the same size as the output */
	this->bp_forward_feed_units[layer_count] = (double *) malloc(this->n_out * sizeof(double));
	
	this->bp_upper_derivatives = (double *) malloc (max_hidden_layer_sz * sizeof(double));
	this->bp_curr_derivatives = (double *) malloc (max_hidden_layer_sz * sizeof(double));
}

/* implements stochastic gradient descent (not full gradient descent) */
/* this function will look like a combination of pre_training and compute_output */
/* input in an array with n_in + 1 values, where the first value is 1. This way vector multiplication is easier to code.*/
void deep_neural_network::backpropagation(double * input, double * target)
{
	if (abs(input[0] - 1) > 0.0000001)
	{
		printf ("the first value in the input should be 1. Back propagation failed.\n");
		return;
	}
	
	/* The algorithm is: */
	/* 1. Calculate the output of the neural network */
	/*    As we go through the layers, save intermediate results */
	/*    We will need these results when we calculate the weights in the */
	/*    back propagation step.*/
	/* 2. Once we calculated the output of the neural network */
	/*    we back track and calculate the derivatives at each step */
	/*    We then update the weights based on the derivatives */
	
	/* 1. Calculate the output of the neural network via forward feed */
	
	this->set_memory_for_backpropagation();
	int calc_iterations  = this->n_hidden_layers + 1;		/* + 1 because the output of the network is after the last hidden layer */
	
	/* cache some pointers for the loop below */
	double ** ff = this->bp_forward_feed_units;
	double ** hw = this->hidden_layers_weights;
	int * ls = this->hidden_layers_sizes;
	for (int i = 0; i < calc_iterations; i ++)
	{
		/* the output of the neural network is a bunch of staggered vector multiplications */
		/* between the current layer of inputs and the current layer of weights */
		double * curr_input;
		double * curr_weights;
		double * curr_res = ff[i];

		int input_size;
		if (i == 0)
		{
			curr_input = input;
			input_size = this->n_in + 1;		/* + 1 because the first input element corresponds to the bias weight and is 1 */
		}
		else
		{
			/* the output from the prior step is now input into the current step */
			curr_input = ff[i-1];
			input_size = ls[i-1];
		}
		curr_weights = hw[i];
		int res_sz = (i == calc_iterations - 1) ? this->n_out : ls[i];
		for (int j = 0; j < res_sz; j ++)
		{
			double tmp = this->vector_multiplication(curr_input, curr_weights + input_size * j, input_size);
			tmp = tmp / this->adjust_for_squashing;
			tmp = this->squashing_function(tmp);
			curr_res[j] = tmp;
		}
	}
	
	/* 2. Once we calculated the output of the neural network */
	/*     we back track and calculate the derivatives at each step */
	/* 3.  We then update the weights based on the derivatives */
	
	/* 2.Calculate the derivative at the current level */
	calc_iterations = this->n_hidden_layers + 1;			/* we iterate on each layer in the hidden layers weights */
	/* cache some pointers for the loops below */
	ff = this->bp_forward_feed_units;
	hw = this->hidden_layers_weights;
	double ** pdd = this->prior_descent_hidden_layers_deltas;
	ls = this->hidden_layers_sizes;
	for (int i = calc_iterations; i > 0; i --)
	{
		/* loop invariants: */
		/* i 		= upper level (prior calculation). Derivatives at upper level calculated at the prior step */
		/* i - 1 	= current level. We are calculating derivatives at this level */
		/* i - 2 	= lower level. These are inputs into the gradient descent */

		double * curr_derivatives_arr = this->bp_curr_derivatives;
		int derivative_arr_sz;
		/* 2.a Calculate the derivative at the current level */
		if (i == calc_iterations)
		{
			/* top layer */
			double * output_arr;
			derivative_arr_sz = this->n_out;
			output_arr = ff[i-1];
			for (int j = 0; j < derivative_arr_sz; j++)
			{
				double tmp = (target[j] - output_arr[j]);
				tmp = this->squashing_function_derivative(output_arr[j]) * tmp;
				tmp = tmp / this->adjust_for_squashing;
				curr_derivatives_arr[j] = tmp;
			}
		}
		else
		{
			/* not the top layer. use chain rule: */
			/* current layer derivative = upper layer weights x upper layer derivative x derivative at the current layer */
			
			/* upper_derivative_arr and upper_weights have the same size */
			int upper_derivative_arr_sz = (i == calc_iterations - 1) ? this->n_out : ls[i];
			double * upper_derivative_arr = this->bp_upper_derivatives;
			double * upper_weights = hw[i];
			
			double * curr_layer = ff[i-1];
			
			/* size of current derivative array */
			derivative_arr_sz = ls[i-1];
			
			/* chain rule */
			for (int j = 0; j < derivative_arr_sz; j++)
			{
				/* upper layer weights x upper layer derivative */
				double tmp = this->vector_multiplication(upper_derivative_arr, upper_weights + j * upper_derivative_arr_sz, upper_derivative_arr_sz);
				/* x derivative at the current layer */
				tmp = this->squashing_function_derivative(curr_layer[j]) * tmp;
				tmp = tmp / this->adjust_for_squashing;
				
				curr_derivatives_arr[j] = tmp;
			}
		}
		
		/* 3. Take a small step on the derivative at the current level */
		double * curr_weights = hw[i-1];
		double * inp = (i == 1) ? input : ff[i-2];
		double * prior_descent_weights = pdd[i-1];
		int first_dim = (i == 1) ? this->n_in + 1 : ls[i-2];
		int sec_dim = derivative_arr_sz;
		double learn_rate = this->learning_rate;
		double moment = this->momentum;
		
		for (int j = 0; j < first_dim; j++)
		{
			for (int k = 0; k < sec_dim; k++)
			{
				double delta = learn_rate * curr_derivatives_arr[k] * inp[j];
				delta += moment * prior_descent_weights[j * sec_dim + k];
				curr_weights[j + k * first_dim] += delta;
				prior_descent_weights[j + k * first_dim] = delta;
			}
		}
		
		/* current derivatives become prior derivatives for the following iteration. */
		/* we don't want to allocate nor do we want to copy memory from curr_derivatives into the prior derivative. */
		/* so we just swap the pointers. */
		double * swp_ptr = this->bp_upper_derivatives;
		this->bp_upper_derivatives = this->bp_curr_derivatives;
		this->bp_curr_derivatives = swp_ptr;
	}
}

/* The memory for the results is pre-allocated and passed into the function via the output parameter.*/
void deep_neural_network::compute_output(double * input, double * output)
{
	/* +1 because layer_to_calc in compute_output_pre_train is 1-based, not 0-based */
	this->compute_output_pre_train(input, this->n_hidden_layers + 1, output);
}

/* The memory for the results is pre-allocated and passed into the function via the output parameter.*/
/* assume the first number in the input is 1, to make it easier to multiply vectors*/
void deep_neural_network::compute_output_pre_train(double * input, int layer_to_calc, double * output)
{
	if (abs(input[0] - 1.0) > 0.0000001)
	{
		printf ("the first number in the input array should be 1. Computation failed.\n");
		return;
	}
	
	/* layer_to_calc is 1-based, not 0-based */
	if (layer_to_calc > this->n_hidden_layers + 1)
	{
		printf ("we can calculate at most %d layers deep. Computation failed.\n", this->n_hidden_layers+1);
		return;
	}
	
	/* cache some pointers */
	double ** hw = this->hidden_layers_weights;
	int * ls = this->hidden_layers_sizes;
	
	double * inp_arr;
	int inp_sz;

	/* if we calculate only one layer, we copy the results directly into the output vector */
	double * res;
	int res_sz = ls[0];
	/* layer_to_calc is 1-based, not 0-based */
	if (layer_to_calc == 1)
	{
		res = output;
	}
	else
	{
		res = this->grow_array(this->compute_output_res, this->compute_output_res_sz, res_sz);
		this->compute_output_res = res;
		if (res_sz > this->compute_output_res_sz)
		{
			this->compute_output_res_sz = res_sz;
		}
	}
	
	/* first layer calculation */
	double * weights = hw[0];
	inp_arr = input;
	inp_sz = this->n_in + 1;
	for (int i = 0; i < res_sz; i++)
	{
		double tmp = this->vector_multiplication(inp_arr, weights + i * inp_sz, inp_sz);
		tmp = tmp / this->adjust_for_squashing;
		tmp = this->squashing_function(tmp);
		res[i] = tmp;
	}
	
	if (layer_to_calc == 1)
	{
		return;
	}
	
	/* calculate the intermediate layers */
	/* cache some pointers */
	hw = this->hidden_layers_weights;
	ls = this->hidden_layers_sizes;
	for (int i = 1; i < layer_to_calc-1; i++)
	{
		/* current results are inputs into the calculation and thus become prior results */
		double * swp_ptr = this->compute_output_res;
		this->compute_output_res = this->compute_output_prior_res;
		this->compute_output_prior_res = swp_ptr;
		int swp_sz = this->compute_output_res_sz;
		this->compute_output_res_sz = this->compute_output_prior_res_sz;
		this->compute_output_prior_res_sz = swp_sz;
		
		int res_sz = ls[i];
		this->compute_output_res = this->grow_array(this->compute_output_res, this->compute_output_res_sz, res_sz);
		if (res_sz > this->compute_output_res_sz)
		{
			this->compute_output_res_sz = res_sz;
		}
		
		weights = hw[i];
		inp_arr = this->compute_output_prior_res;
		double * res = this->compute_output_res;
		inp_sz = ls[i-1];
		for (int j = 0; j < res_sz; j++)
		{
			double tmp = this->vector_multiplication(inp_arr, weights + j * inp_sz, inp_sz);
			tmp = tmp / this->adjust_for_squashing;
			tmp = this->squashing_function(tmp);
			res[j] = tmp;
		}
	}
	
	/* last layer to calculate. Copy the results into the output buffer */
	/* cache some pointers */
	hw = this->hidden_layers_weights;
	ls = this->hidden_layers_sizes;
	weights = hw[layer_to_calc-1];
	/* current results are inputs into the calculation */
	inp_arr = this->compute_output_res;
	inp_sz = ls[layer_to_calc-2];
	res_sz =  (layer_to_calc == this->n_hidden_layers+1) ? this->n_out : ls[layer_to_calc-1];
	for (int i = 0; i < res_sz; i ++)
	{
		double tmp = this->vector_multiplication(inp_arr, weights + i * inp_sz, inp_sz);
		tmp = tmp / this->adjust_for_squashing;
		tmp = this->squashing_function(tmp);
		output[i] = tmp;
	}
}

void deep_neural_network::dispose_threads()
{
	if (this->thread_count == 1)
	{
		return;
	}
	
	for (int i = 0; i < this->thread_count; i++)
	{
		/* should exit_thread = true be atomic???? */
		this->thread_infos[i].exit_thread = true;
		ReleaseSemaphore(this->thread_infos[i].wait_for_work, 1, NULL /*we do not need the previous count*/);
	}
	
	WaitForMultipleObjects(this->thread_count, this->threads, true /*wait all*/, INFINITE /*wait until the threads exit*/);

	for (int i = 0; i < this->thread_count; i++)
	{
		CloseHandle(this->thread_infos[i].wait_for_work);
		CloseHandle(this->thread_infos[i].work_done);
		CloseHandle(this->threads[i]);
	}
	
	free (this->thread_infos);
	free (this->threads);
}

/* the worker thread for multi-threaded vector multiplication */
DWORD WINAPI deep_neural_network::worker_thread(void * in)
{
	thread_info * ti = (thread_info *) in;
	
	while (true)
	{
		WaitForSingleObject(ti->wait_for_work, INFINITE);
		/* all done*/
		if (ti->exit_thread)
		{
			return 0;
		}
		
		/* here is what we are here to do */
		double r = 0.0;
		double * in_1 = ti->v1;
		double * in_2 = ti->v2;
		int sz = ti->v_sz;
		for (int i = 0; i < sz; i++)
		{
			r += in_1[i] * in_2[i];		
		}
		ti->res = r;
		
		ReleaseSemaphore(ti->work_done, 1, NULL /*we don't need the previous count*/);
	}
}

double deep_neural_network::vector_multiplication(double * in_1, double * in_2, int len)
{
	/*
	 * At what point is the multi-threaded overhead (spawning threads, waiting on semaphores) overcame by
	 * size of the multiplication? Tests on my machine indicate len at about 7,500. So if we multiply
	 * less than 10,000 numbers just do it single threaded.
	 */
	if (this->thread_count == 1 || (len < 10000))
	{
		/* single threaded or not enough work to do */
		double res = 0.0;
		for (int i = 0; i < len; i ++)
			res += in_1[i] * in_2[i];
		
		return res;
	}
	else		
	{
		/* multi threaded */
		int chunk = len / this->thread_count;
		int t_c = this->thread_count;
		thread_info * t_i = this->thread_infos;

		/* split the work */
		/* start the work */
		int start = 0;
		for (int i = 0; i < t_c ; i ++)
		{
			thread_info * ti = t_i + i;
			ti->v1 = in_1 + start;
			ti->v2 = in_2 + start;
			if (chunk + start > len)
			{
				chunk = len - start;
			}
			ti->v_sz = chunk;
			ReleaseSemaphore(ti->wait_for_work, 1, NULL /*we don't need the previous count */);
			start += chunk;
		}
		
		/* wait for the results */
		for (int i = 0; i < t_c ; i++)
		{
			thread_info * ti = t_i + i;
			WaitForSingleObject(ti->work_done, INFINITE);
		}
		
		/* now calculate the results */
		double res = 0.0;
		for (int i = 0; i < t_c ; i ++)
		{
			thread_info * ti = t_i + i;
			res += ti->res;
		}
		
		return res;
	}
}

double deep_neural_network::squashing_function(double in)
{
	/*
	 * Use sigmoid function
	 */
	
	if (this->nn_type == SIGMOID)
	{
		return 1 / (1 + exp(0 - in));
	}
	else
	{
		/* gotta be the RELU */
		if (in > 0)
		{
			return in;
		}
		else
		{
			return 0;
		}
	}
}

double deep_neural_network::squashing_function_derivative(double func_value)
{
	/*
	 * We will use the sigmoid function for squashing.
	 * The derivative of the sigmoid(x) function is sigmoid(x) * (1-sigmoid(x))
	 * In our notation, func_value = sigmoid(x).
	 */
	if (this->nn_type == SIGMOID)
	{
		return func_value * (1-func_value);
	}
	else
	{
		/* gotta be the RELU */
		if (func_value > 0)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}
}

void deep_neural_network::save_to_file(char * file_name)
{
	
	/* the file format is the following: */
	/* first two bytes are "DH" */
	/* then 4 bytes for n_in written as an integer */
	/* then 4 bytes for n_out written as an integer */
	/* then 4 bytes for number of hidden layers written as an integer */
	/* then 4 bytes for the size of each hidden layers, each written as an integer */
	/* then learning_rate */
	/* then momentum */
	/* then the type (as an integer ) */
	/* then thread count */
	/* then the weights */
	FILE * out = fopen(file_name, "wb");
	
	if (!out)
	{		
		return;
	}

	/* the file format is the following: */
	/* first two bytes are "DH" */
	fwrite("DH", sizeof(char), 2, out);

	/* then 4 bytes for n_in written as an integer */
	fwrite( (char *) (& this->n_in) /*ptr*/, sizeof(int) /*size*/, 1 /*count*/, out /*stream*/);
	
	/* then 4 bytes for n_out written as an integer */
	fwrite( (char *) (& this->n_out) /*ptr*/, sizeof(int) /*size*/, 1 /*count*/, out /*stream*/);
	
	/* then 4 bytes for number of hidden layers written as an integer */
	fwrite( (char *) (& this->n_hidden_layers) /*ptr*/, sizeof(int) /*size*/, 1 /*count*/, out /*stream*/);
	
	/* then 4 bytes for the size of each hidden layers, each written as an integer */
	for (int i = 0; i < this->n_hidden_layers; i++)
	{
		fwrite( (char *) (this->hidden_layers_sizes + i) /*ptr*/, sizeof(int) /*size*/, 1 /*count*/, out /*stream*/);

	}
	/* then learning_rate */
	fwrite( (char *) (& this->learning_rate) /*ptr*/, sizeof(double) /*size*/, 1 /*num*/, out /*stream*/);
	/* then momentum */
	fwrite( (char *) (& this->momentum) /*ptr*/, sizeof(double) /*size*/, 1 /*num*/, out /*stream*/);
	/* then the type (as an integer ) */
	fwrite( (char *) (& this->nn_type) /*ptr*/, sizeof(int) /*size*/, 1 /*num*/, out /*stream*/);

	/* then thread_count */
	fwrite( (char *) (& this->thread_count) /*ptr*/, sizeof(int) /*size*/, 1 /*num*/, out /*stream*/);

	/* then the weights */
	/* cache some pointers */
	int * layer_sizes = this->hidden_layers_sizes;
	double ** weights = this->hidden_layers_weights;
	/* first the weights from the inputs to the first hidden layer */
	int sz = (this->n_in + 1) * layer_sizes[0];
	fwrite((char *) (weights[0]) /*ptr*/, sizeof(double) /*size*/, sz /*num*/, out /*stream*/);
	/* now the intermediary weights */
	for (int i = 1; i < this->n_hidden_layers; i++)
	{
		sz = layer_sizes[i-1] * layer_sizes[i];
		fwrite((char *) (weights[i]) /*ptr*/, sizeof(double) /*size*/, sz /*num*/, out /*stream*/);
	}
	/* now the last layer (from the weights to the output) */
	sz = layer_sizes[this->n_hidden_layers-1] * this->n_out;
	fwrite((char *) (weights[this->n_hidden_layers]) /*ptr*/, sizeof(double) /*size*/, sz /*num*/, out /*stream*/);
	fclose(out);
}

deep_neural_network * deep_neural_network::read_from_file(char * file_name)
{
	FILE * in = fopen(file_name, "rb");
	if (!in)
	{
		return NULL;
	}
	/* neural network initialization constants */
	int n_in, n_out, n_hidden_layers, thread_count;
	nn_type_enum nn_type;
	double learning_rate, momentum;
	
	/* 64 hidden layers should be enough for anyone */
	int MAX_LAYERS = 64;
	int layers_sizes[MAX_LAYERS];
	
	/* the buffer in which we will read over and over again */
	char buf[64];
	
	/* the file format is the following: */
	/* first two bytes are "DH" */
	/* then 4 bytes for n_in written as an integer */
	/* then 4 bytes for n_out written as an integer */
	/* then 4 bytes for number of hidden layers written as an integer */
	/* then 4 bytes for the size of each hidden layers, each written as an integer */
	/* then learning_rate */
	/* then momentum */
	/* then the type */
	/* then thread count */
	/* then the weights */
	
	/* the file format is the following: */
	/* first two bytes are "DH" */
	fread(buf /*dest*/, sizeof(char) /*size*/, 2 /*count*/, in);
	if (buf[0] != 'D' || buf[1] != 'H')
	{
		return NULL;
	}
	
	/* then 4 bytes for n_in written as an integer */
	fread( (char *) (&n_in) /*dest*/, sizeof(int) /*size*/, 1 /*count*/, in);
	/* then 4 bytes for n_out written as an integer */
	fread( (char *) (&n_out) /*dest*/, sizeof(int) /*size*/, 1 /*count*/, in);
	/* then 4 bytes for number of hidden layers written as an integer */
	fread( (char *) (&n_hidden_layers) /*dest*/, sizeof(int) /*size*/, 1 /*count*/, in);
	/* then 4 bytes for the size of each hidden layers, each written as an integer */
	for (int i = 0; i < n_hidden_layers; i ++)
	{
		fread( (char *) (layers_sizes + i) /*dest*/, sizeof(int) /*size*/, 1 /*count*/, in);
	}
	
	/* then learning_rate */
	fread( (char *) (&learning_rate) /*dest*/, sizeof(double) /*size*/, 1 /*count*/, in);
	/* then momentum */
	fread( (char *) (&momentum) /*dest*/, sizeof(double) /*size*/, 1 /*count*/, in);
	/* then the neural network type */
	fread( (char *) (&nn_type) /*dest*/, sizeof(int) /*size*/, 1 /*count*/, in);
	/* then thread count */
	fread( (char *) (&thread_count) /*dest*/, sizeof(int) /*size*/, 1 /*count*/, in);
	
	/* create the neural network */
	deep_neural_network * res = new deep_neural_network(n_in, n_out, learning_rate, momentum, n_hidden_layers, layers_sizes, thread_count, nn_type);
	
	/* then the weights */
	double ** weights = (double **) malloc ( (n_hidden_layers + 1) * sizeof(double));
	/* first weights: from inputs to the nn */
	int sz = (n_in + 1) * layers_sizes[0];
	weights[0] = (double *) malloc ( sz * sizeof(double));
	fread( (char *) (weights[0]) /*dest*/, sizeof(double) /*size*/, sz /*count*/, in);
	
	/* intermediary weights */
	for (int i = 1; i < n_hidden_layers; i ++)
	{
		int sz = layers_sizes[i-1] * layers_sizes[i];
		weights[i] = (double *) malloc ( sz * sizeof(double));
		fread( (char *) (weights[i]) /*dest*/, sizeof(double) /*size*/, sz /*count*/, in);
	}
	
	/* last weights: from nn to the outputs */
	sz = layers_sizes[n_hidden_layers-1] * n_out;
	weights[n_hidden_layers] = (double *) malloc ( sz * sizeof(double));
	fread( (char *) (weights[n_hidden_layers]) /*dest*/, sizeof(double) /*size*/, sz /*count*/, in);
	
	res->hidden_layers_weights = weights;
	
	fclose(in);
	
	return res;
}

bool deep_neural_network::identical_neural_networks(deep_neural_network * cmp)
{
	if (this->n_in != cmp->n_in)
		return false;
	
	if (this->n_out != cmp->n_out)
		return false;
	
	if (abs(this->learning_rate - cmp->learning_rate) > 0.00000001)
		return false;
	
	if (abs(this->momentum - cmp->momentum) > 0.00000001)
		return false;
	
	if (this->n_hidden_layers != cmp->n_hidden_layers)
		return false;
	
	for (int i = 0; i < this->n_hidden_layers; i ++)
	{
		if (this->hidden_layers_sizes[i] != cmp->hidden_layers_sizes[i])
		{
			return false;
		}
	}
	
	/* check the first layer */
	for (int i = 0; i < (this->n_in + 1)*(this->hidden_layers_sizes[0]); i++)
	{
		double diff = *(this->hidden_layers_weights[0]+i) - *(cmp->hidden_layers_weights[0]+i);
		if (abs(diff) > 0.00000001 )
		{
			return false;
		}
	}
	
	/* check the rest of the layers */
	for (int i = 1; i < this->n_hidden_layers; i ++)
	{
		for (int j = 0; j < this->hidden_layers_sizes[i] * this->hidden_layers_sizes[i-1]; j ++)
		{
			double diff = *(this->hidden_layers_weights[i] + j) - *(cmp->hidden_layers_weights[i] + j);
			if (abs(diff) > 0.00000001 )
			{
				return false;
			}
		}
	}
	
	/* check the last layer (the one that outputs) */
	for (int j = 0; j < this->hidden_layers_sizes[this->n_hidden_layers-1] * this->n_out; j++)
	{
		double diff = *(this->hidden_layers_weights[this->n_hidden_layers]+j) - *(cmp->hidden_layers_weights[cmp->n_hidden_layers]+j);
		if (abs(diff) > 0.00000001 )
		{
			return false;
		}
	}

	return true;
}

void deep_neural_network::fake_back_propagation()
{
	/* initialize the weights to some  numbers */
	/* assume that n_in, n_out, hidden_layers etc... are defined and all memory is allocated */
	
	for (int i = 0; i < this->n_hidden_layers+1; i++)
	{
		double * current_layer = this->hidden_layers_weights[i];
		int first_dim, second_dim;
		if (i==0)
		{
			first_dim = this->n_in + 1;
			second_dim = this->hidden_layers_sizes[i];
		}
		else
		{
			first_dim = this->hidden_layers_sizes[i-1];
			if (i == this->n_hidden_layers)
			{
				second_dim = this->n_out;
			}
			else
			{
				second_dim = this->hidden_layers_sizes[i];
			}
		}

		for (int j = 0; j < first_dim; j++)
		{
			for (int k = 0; k < second_dim; k++)
			{
				current_layer[j+k*first_dim] = (double) (0.1 * (j+1) + 0.1*(k+1));
			}
		}
	}
}

double * deep_neural_network::grow_array(double * src, int src_sz, int new_sz)
{
	if (src_sz >= new_sz)
	{
		return src;
	}
	
	if (src)
	{
		free(src);
	}
	
	src = (double *) malloc (new_sz * sizeof(double));
	
	return src;
}

void deep_neural_network::print_neural_network()
{
	printf("neural network printout\n");
	printf("\tn_in: %d\n",this->n_in);
	printf("\tn_out: %d\n",this->n_out);
	printf("\tn_hidden_layers: %d\n",this->n_hidden_layers);
	for (int i = 0; i < this->n_hidden_layers; i++)
	{
		printf("\t\thidden layer %d size: %d\n",i, this->hidden_layers_sizes[i]);
	}
	printf("\tweights\n");
	for (int i = 0; i < this->n_hidden_layers; i++)
	{
		double * current_layer = this->hidden_layers_weights[i];
		int first_dim = (i == 0) ? n_in+1 : this->hidden_layers_sizes[i-1];
		int second_dim = this->hidden_layers_sizes[i];
		int current_layer_size = first_dim * second_dim;
		printf("\tweights for layer %d\n",i);
		for (int j = 0; j < current_layer_size; j++)
		{
			printf("\t\t%f\n",current_layer[j]);
		}
	}
}