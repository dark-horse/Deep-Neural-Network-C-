Multi Layer Perceptron Neural Network

Disclaimer

My friend said I should put this disclaimer:

The program and source code contained in this repository are for general usage purposes only. We make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to program and the source code contained in this repository. Any reliance you place on such information is therefore strictly at your own risk. In no event will we be liable for any loss or damage including without limitation, indirect or consequential loss or damage, or any loss or damage whatsoever arising from loss of data or profits arising out of, or in connection with, the use / misuse / abuse of this program and source code.

[In other words, this is the Internet and you are using something that someone called The Dark Horse uploaded on a free site. Try explaining that to your boss / in-laws/ children / etc... if things don't go smoothly]

Now for the fun part.

This code is released into the public domain. Feel free to use / change as you see fit this code or parts of it.

This program is written by a hobbyist to learn / play with machine learning in C++.

The program reads uses the MNIST database of hand written digits to train and then tests the neural network.
The neural network implements two types of activation (SIGMOID and RELU).
This version of MNIST database contains all the images in one large file in the portable gray format; and all the labels in a different file.
There are other versions of the MNIST database (for example, the image for each hand written is in its own bitmap file).

I tested the neural network under the following conditions:
1. Shallow net with 100 hidden nodes trained with 30 descents through the training data. Activation method is SIGMOID. This neural network achieved 94% accuracy on both training and test data sets. It took about 60 seconds to do a full descent (i.e. backpropagation for all 60,000 images in the training set).
2. Shallow net with 300 hidden nodes using SIGMOID activation. Somewhat surprisingly, this neural network had lower performance. The best the 300 hidden nodes neural network did was 90% accuracy after 60 training descents. It took about 4 minutes to do backpropagation for all 60,000 images in the training set.
3. SIGMOID 2-layer net: first layer has 500 hidden nodes, second layer has 100 nodes. After 180 descends for the first layer and 30 for the second layer this neural network performance peaked at 84% accuracy.
4. RELU 2-layer net: first layer has 500 hidden nodes, second layer has 100 nodes. After 180 descends for the first layer and 30 for the second layer this neural network performance peaked at 95% accuracy.

So far it seems (somewhat counter-intuitively) that a shallow net neural network is the most accurate.

Hope you find this code useful.

