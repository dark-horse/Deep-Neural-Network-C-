/*
 * Comment
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "deep_neural_network.h"


int main(int argc, char * argv[])
{
	
	/* neural machine init stuff */
	int input_size = 1748;
	int output_size = 300;
	double learning_rate = 0.3;
	double momentum = 0.3;
	int layers_count = 3;
	int n_layers_deep[3];
	n_layers_deep[0] = 500;
	n_layers_deep[1] = 100;
	n_layers_deep[2] = 50;
	deep_neural_network * dnn = new deep_neural_network(input_size, output_size, learning_rate, momentum, layers_count, n_layers_deep, 3);
	dnn->init();
	
	/* test write and read */
	dnn->fake_back_propagation();
	
	double * input = (double *) malloc ((input_size + 1 ) * sizeof(double));
	input[0] = 1.0;
	for (int i = 1; i < input_size+1; i++)
	{
		input [i] = 0.00001 * i - 0.001 * i * i;
	}
	
	double * output = (double *) malloc (output_size * sizeof(double));
	
	dnn->compute_output(input, output);
	
	for (int i = 0; i < output_size; i++)
	{
		printf("output %d is %f.\n", i, output[i]);
	}
	
	if (dnn)
	{
		dnn->dispose();
		delete(dnn);
		dnn = NULL;
	}
	
	if (input)
	{
		free(input);
		input = NULL;
	}
	
	if (output)
	{
		free (output);
		output = NULL;
	}


	return 1;
}
