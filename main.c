 /*
 * Comment
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "deep_neural_network.h"

double zero_v[10] = 	{0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
double one_v[10] = 		{0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
double two_v[10] = 		{0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
double three_v[10] = 	{0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
double four_v[10] = 	{0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1};
double five_v[10] = 	{0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1};
double six_v[10] = 		{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1};
double seven_v[10] =	{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1};
double eight_v[10] = 	{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1};
double nine_v[10] = 	{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9};

double * digits[10];

void print_help();
int parse_cmd_ln(int argc, char * argv[], char ** path, int * path_sz, int * n_hidden_layer, int * hidden_layers_sizes, int * thread_count, nn_type_enum * nn_type);
void parse_img_file_hdr(FILE * imgs_f, int * imgs_count, int * img_w, int *img_h);
void parse_lbl_file_hdr(FILE * lbls_f, int * lbls_count);
void pre_train_neural_network(deep_neural_network * dnn, int imgs_count, int dnn_n_in, int layers_count, FILE * imgs_f);
void calculate_training_layers_error(deep_neural_network * dnn, int imgs_count, int input_size, int layers_count, int first_layer_size, FILE * imgs_f);
void back_propagation_neural_network(deep_neural_network * dnn, int imgs_count, int dnn_n_in, FILE * imgs_f, FILE * lbls_f);
void test_neural_network(deep_neural_network * dnn, FILE * tst_imgs_f, FILE * tst_lbls_f);

bool identical_vectors(double * in1, double * in2, int len)
{
	double max_in1, max_in2;
	int max_in1_index, max_in2_index;
	
	max_in1 = -1000000.0;
	max_in2 = -1000000.0;
	
	max_in1_index = 0;
	max_in2_index = 0;
	
	for (int i = 0; i < len; i ++)
	{
		if (max_in1 < in1[i])
		{
			max_in1 = in1[i];
			max_in1_index = i;
		}
		
		if (max_in2 < in2[i])
		{
			max_in2 = in2[i];
			max_in2_index = i;
		}
	}
	
	return (max_in1_index == max_in2_index);
}

int main(int argc, char * argv[])
{
	/* before parsing the command line, pick some default choices for the inputs */
	double learning_rate = 0.3;
	double momentum = 0.3;
	int layers_count = 1;
	nn_type_enum nn_type = SIGMOID;
	int n_layers_deep[16];

	/* default hidden layer size is 100 */
	memset (n_layers_deep, 100, sizeof(int) * 16);
	int thread_count = 1;
	
	int path_sz;
	char * path;
	
	if (!parse_cmd_ln(argc, argv, &path, &path_sz, &layers_count, n_layers_deep, &thread_count, &nn_type))
	{
		return 1;
	}
	
	printf ("path is : %s.\n", path);
	printf ("path size is %d.\n", path_sz);
	printf ("layers_count is %d.\n", layers_count);
	for (int i = 0; i < layers_count; i ++)
	{
		printf("\t layer count %d has size %d.\n", i, n_layers_deep[i]);
	}
	printf ("thread count is: %d.\n", thread_count);
	if (nn_type == SIGMOID)
	{
		printf("nn type is SIGMOID.\n");
	}
	else if (nn_type == RELU)
	{
		printf("nn type is RELU.\n");
	}
	else
	{
		printf("nn type not defined.\n");
	}
		
	digits[0] = zero_v;
	digits[1] = one_v;
	digits[2] = two_v;
	digits[3] = three_v;
	digits[4] = four_v;
	digits[5] = five_v;
	digits[6] = six_v;
	digits[7] = seven_v;
	digits[8] = eight_v;
	digits[9] = nine_v;
	
	char train_imgs_fn[] = "train-images-idx3-ubyte";
	char train_lbls_fn[] = "train-labels-idx1-ubyte";
	char test_imgs_fn[] = "t10k-images-idx3-ubyte";
	char test_lbls_fn[] = "t10k-labels-idx1-ubyte";
	char * imgs_path = (char *) malloc (path_sz + 1 + strlen(train_imgs_fn) + 1);
	char * lbls_path = (char *) malloc (path_sz + 1 + strlen(train_lbls_fn) + 1);
	
	memcpy (imgs_path /*dest*/, path /*src*/, path_sz /*num*/);
	imgs_path[path_sz] = '\\';
	memcpy (imgs_path + path_sz +1 /*dest*/, train_imgs_fn /*src*/, strlen(train_imgs_fn) /*num*/);
	imgs_path[path_sz + 1 + strlen(train_imgs_fn)] = '\0';
	
	memcpy (lbls_path /*dest*/, path /*src*/, path_sz /*num*/);
	lbls_path[path_sz] = '\\';
	memcpy (lbls_path + path_sz + 1/*dest*/, train_lbls_fn /*src*/, strlen(train_lbls_fn) /*num*/);
	lbls_path[path_sz + 1 + strlen(train_lbls_fn)] = '\0';

	FILE * imgs_f = fopen(imgs_path, "rb");
	FILE * lbls_f = fopen(lbls_path, "rb");
	if (imgs_f == NULL)
	{
		printf ("incorrect file path for images file: %s. Exiting.\n",imgs_path);
	}
	
	if (lbls_f == NULL)
	{
		printf ("incorrect file path for labels files: %s. Exiting.\n", lbls_path);
	}
	
	int imgs_count, img_w, img_h, lbls_count;
	parse_img_file_hdr(imgs_f, &imgs_count, &img_w, &img_h);
	parse_lbl_file_hdr(lbls_f, &lbls_count);
	
	/* neural machine init stuff */
	int input_size = img_w * img_h;
	int output_size = 10;

	deep_neural_network * dnn = new deep_neural_network(input_size, output_size, learning_rate, momentum, layers_count, n_layers_deep, thread_count, nn_type);
	dnn->init();
	
	pre_train_neural_network(dnn, imgs_count, input_size, layers_count, imgs_f);
	
	calculate_training_layers_error(dnn, imgs_count, input_size, layers_count, n_layers_deep[0], imgs_f);

	back_propagation_neural_network(dnn, imgs_count, input_size, imgs_f, lbls_f);

	char fn[] = "temp.nn\0";
	dnn->save_to_file(fn);
	
	/* how well the neural network does on the training data */
	test_neural_network(dnn, imgs_f, lbls_f);
	
	/* how well the neural network does on the test data (not training data) */
	memcpy (imgs_path /*dest*/, path /*src*/, path_sz /*num*/);
	imgs_path[path_sz] = '\\';
	memcpy (imgs_path + path_sz +1 /*dest*/, test_imgs_fn /*src*/, strlen(test_imgs_fn) /*num*/);
	imgs_path[path_sz + 1 + strlen(test_imgs_fn)] = '\0';
	
	memcpy (lbls_path /*dest*/, path /*src*/, path_sz /*num*/);
	lbls_path[path_sz] = '\\';
	memcpy (lbls_path + path_sz + 1/*dest*/, test_lbls_fn /*src*/, strlen(test_lbls_fn) /*num*/);
	lbls_path[path_sz + 1 + strlen(test_lbls_fn)] = '\0';

	FILE * tst_imgs_f = fopen(imgs_path, "rb");
	FILE * tst_lbls_f = fopen(lbls_path, "rb");
	
	
	if (tst_imgs_f == NULL)
	{
		printf ("incorrect file path for test images file: %s. Exiting.\n", imgs_path);
	}

	if (tst_lbls_f == NULL)
	{
		printf ("incorrect file path for test labels file: %s. Exiting.\n", lbls_path);
	}
	
	test_neural_network(dnn, tst_imgs_f, tst_lbls_f);
	
	if (dnn)
	{
		dnn->dispose();
		delete(dnn);
		dnn = NULL;
	}

	/* some cleanup */
	if (imgs_f)
		fclose(imgs_f);
	if (lbls_f)
		fclose(lbls_f);
	
	if (tst_imgs_f)
		fclose(tst_imgs_f);
	if (tst_lbls_f)
		fclose (tst_lbls_f);
	if (imgs_path)
		free(imgs_path);
	if (lbls_path)
		free(lbls_path);

	return 1;
}


void print_help(char * program_name)
{
	printf ("Usage: %s [options] ...\n", program_name);
	printf ("Options:\n");
	printf ("  -h \tPrint this message and exit.\n");
	printf ("  -i \tThe input path (the directory containing the training and test pictures and labels. Not optional - path must be specified.\n");
	printf ("  -t \tThe type of the neural network. Can be RELU or SIGMOID. Optional - if not defined, default is SIGMOID.\n");
	printf ("  -n \tThe number of hidden layers in the neural network. Followed by the size of the layers. At most 16 layers.\n");
	printf ("     \tOptional - if not defined, default is 1 (shallow net) and size of first layer is 100.\n");
	printf ("  -c \tThe number of threads in the neural network. Optional - if not defined, default is 1.\n");
}

/* if command line has an error, print the help message and return 0. other wise fill in the arguments and return 1 */
int parse_cmd_ln(int argc, char * argv[], char ** path, int * path_sz, int * n_hidden_layer, int * hidden_layers_sizes, int * thread_count, nn_type_enum * nn_type)
{
	/* the only necessary argument is the path: -i */
	bool path_specified = false;
	
	/* the position in the argument list */
	int pos = 1;
	
	while (pos < argc)
	{
		if (strncmp(argv[pos], "-h", 2) == 0)
		{
			print_help(argv[0]);
			/* not quite error but we have to exit the program anyway */
			return 0;
		}
		else if (strncmp(argv[pos], "-i", 2) == 0)
		{
			pos ++;
			* path = argv[pos];
			* path_sz = strlen(argv[pos]);
			path_specified = true;
			pos ++;
		}
		else if (strncmp(argv[pos], "-t", 2) == 0)
		{
			pos ++;
			if (strncmp(argv[pos], "RELU", 4) == 0)
			{
				*nn_type = RELU;
			}
			else if (strncmp(argv[pos], "SIGMOID", 7) == 0)
			{
				*nn_type = SIGMOID;
			}
			else
			{
				/* ERROR */
				print_help(argv[0]);
				return 0;
			}
			pos ++;
		}
		else if (strncmp(argv[pos], "-n", 2) == 0)
		{
			pos ++;
			*n_hidden_layer = atoi(argv[pos]);
			pos ++;
			for (int i = 0; i < *n_hidden_layer; i++)
			{
				hidden_layers_sizes[i] = atoi(argv[pos]);
				pos ++;
			}
		}
		else if (strncmp(argv[pos], "-c", 2) == 0)
		{
			* thread_count = atoi(argv[pos+1]);
			pos += 2;
		}
	}
	
	if (!path_specified)
	{
		/* ERROR */
		print_help(argv[0]);
		return 0;
	}
	else
	{
		return 1;
	}
	
}

void parse_img_file_hdr(FILE * imgs_f, int * imgs_count, int * img_w, int *img_h)
{
	/* reset the file pointer */
	fseek (imgs_f, (long int) 0, (int) SEEK_SET);
	
	/* first 4 bytes in the imgs_file should be equal to 2051. note that imgs_file is in big-endian mode. */
	int magic_number = 0;
	for (int i = 0; i < 4; i ++)
	{
		magic_number = (magic_number << 8) | fgetc(imgs_f);
	}
	printf ("magic number for images file is %d.\n",magic_number);

	*imgs_count = 0;
	for (int i = 0; i < 4; i ++)
	{
		*imgs_count = (*imgs_count << 8) | fgetc(imgs_f);
	}
	printf("number of images is %d.\n", *imgs_count);

	*img_w = 0;
	for (int i = 0; i < 4; i ++)
	{
		*img_w = (*img_w << 8) | fgetc(imgs_f);
	}

	*img_h = 0;
	for (int i = 0; i < 4; i ++)
	{
		*img_h = (*img_h << 8) | fgetc(imgs_f);
	}
	printf("image width is %d and height is %d.\n", *img_w, *img_h);
}

void parse_lbl_file_hdr(FILE * lbls_f, int * lbls_count)
{
	/* reset the file pointer */
	fseek (lbls_f, (long int) 0, (int) SEEK_SET);

	int magic_number = 0;
	for (int i = 0; i < 4; i ++)
	{
		magic_number = (magic_number << 8) | fgetc(lbls_f);
	}
	printf ("magic number for labels file is %d.\n", magic_number);
	
	*lbls_count = 0;
	for (int i = 0; i < 4; i ++)
	{
		*lbls_count = (*lbls_count << 8) | fgetc(lbls_f);
	}
	printf ("number of labels is %d.\n", *lbls_count);
}

void pre_train_neural_network(deep_neural_network * dnn, int imgs_count, int dnn_n_in, int layers_count, FILE * imgs_f)
{
	/* how many iterations of traying for each of the layers */
	/* one iteration means going through all the images in the images file */
	int train_count = 90;
	double * input = (double *) malloc ((dnn_n_in + 1) * sizeof(double));
	for (int k = 0; k < layers_count; k ++)
	{
		for (int i = 0; i < train_count; i ++)
		{
			/* run 90 iterations on first layer and 30 on the second */
			if (k>0 && i > 29) break;
			int start = time(NULL);
			
			fseek (imgs_f, (long int) 0 + 4 + 4 + 4 + 4, (int) SEEK_SET);

			for (int j = 0; j < imgs_count; j ++)
			{
				/* initialize the input */
				input [0] = 1.0;
				for (int n = 1; n < dnn_n_in + 1; n++)
				{
					unsigned char c;
					fread(&c /*dest*/, 1 /*size*/, 1 /*count*/, imgs_f);
					input[n] = ((double)(c - '\0')) / 1000;
				}
				/* unsupervised training */
				dnn->pre_train(input, k + 1 /*k is the layer we are currently training. Index into the Pretrain method is 1-based */);
			}

			int end = time(NULL);
			printf ("pre-training iteration %d for layer %d took %d seconds.\n", i, k + 1, end - start);
		}
	}
	
	free(input);
}

void calculate_training_layers_error(deep_neural_network * dnn, int imgs_count, int input_size, int layers_count, int first_layer_size, FILE * imgs_f)
{
	double * input = (double *) malloc (input_size * sizeof(double));
	
	for (int k = 0; k < layers_count; k++)
	{
		double err = 0.0;
			
		fseek (imgs_f, (long int) 0 + 4 + 4 + 4 + 4, (int) SEEK_SET);

		for (int j = 0; j < imgs_count; j ++)
		{
			/* initialize the input */
			input [0] = 1.0;
			for (int n = 1; n < input_size + 1; n++)
			{
				unsigned char c;
				fread(&c /*dest*/, 1 /*size*/, 1 /*count*/, imgs_f);
				input[n] = ((double)(c - '\0')) / 1000;
			}
			
			err += dnn->compute_pre_train_error(input, k + 1);
		}
		
		int inp_size = (k) ? first_layer_size : input_size + 1;
		
		printf("the error for pre-training layer %d taken over %d inputs with input size %d is %f.\n", k + 1, imgs_count, inp_size, err);
	}
	
	free(input);
}

void back_propagation_neural_network(deep_neural_network * dnn, int imgs_count, int dnn_n_in, FILE * imgs_f, FILE * lbls_f)
{
	int iter_count = 30;
	double * input = (double *) malloc ((dnn_n_in + 1) * sizeof(double));
	for (int i = 0; i < iter_count; i ++)
	{
		int start = time(NULL);
		
		fseek (imgs_f, (long int) 0 + 4 + 4 + 4 + 4, (int) SEEK_SET);
		fseek (lbls_f, (long int) 0 + 4 + 4, (int) SEEK_SET);

		for (int j = 0; j < imgs_count; j ++)
		{
			/* initialize the input */
			input [0] = 1.0;
			for (int k = 1; k < dnn_n_in + 1; k++)
			{
				unsigned char c;
				fread(&c /*dest*/, 1 /*size*/, 1 /*count*/, imgs_f);
				input[k] = ((double)(c - '\0')) / 1000;
			}
			/* read the target as well */
			unsigned char c;
			fread(&c /* dest */, 1 /*size*/, 1 /*count*/, lbls_f);
			int target_digit = c - '\0';
			if (target_digit > 9 || target_digit < 0)
			{
				printf("some wrong in our file. target digit is %d. skipping this digit.\n",target_digit);
				continue;
			}
			double * target = digits[target_digit];
			
			/* now do the back propagation */
			dnn->backpropagation(input, target);
		}

		int end = time(NULL);
		printf ("iteration %d took %d seconds.\n", i, end - start);
	}
	
	free(input);
}

void test_neural_network(deep_neural_network * dnn, FILE * tst_imgs_f, FILE * tst_lbls_f)
{
	double err = 0.0;
	int correct_answers = 0;

	int imgs_count, img_w, img_h, lbls_count;
	parse_img_file_hdr(tst_imgs_f, &imgs_count, &img_w, &img_h);
	parse_lbl_file_hdr(tst_lbls_f, & lbls_count);
	
	int input_size = img_w * img_h;
	int output_size = 10; /* there are only 10 digits to test */
	
	double * input = (double *) malloc( (input_size + 1) * sizeof(double));
	double output[output_size];

	for (int i = 0; i < imgs_count; i++)
	{
		/* initialize the input */
		input [0] = 1.0;
		for (int k = 1; k < input_size + 1; k++)
		{
			unsigned char c;
			fread(&c /*dest*/, 1 /*size*/, 1 /*count*/, tst_imgs_f);
			input[k] = ((double)(c - '\0')) / 1000;
		}

		/* read the target as well */
		unsigned char c;
		fread(&c /* dest */, 1 /*size*/, 1 /*count*/, tst_lbls_f);
		int target_digit = c - '\0';
		if (target_digit > 9)
		{
			printf("some wrong in our file. target digit is %d. skipping this digit.\n",target_digit);
			continue;
		}
		double * target = digits[target_digit];
		
		dnn->compute_output(input, output);
		
		for (int j = 0; j < output_size; j++)
		{
			err += (target[j] - output[j]) * (target[j] - output[j]);
		}
		
		if (identical_vectors(output, target, output_size))
		{
			correct_answers ++;
		}
	}

	printf("error on the set is: %f. Size of the set is %d.\n", err, imgs_count);
	printf("correct answers on the set is: %d. Size of the set is %d.\n", correct_answers, imgs_count);
	
	free (input);
}