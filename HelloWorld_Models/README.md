# Neural Network Models - Hello World Example

This folder contains the code used to generate the neural network for the **Hello World** example. These files are based upon the [TensorFlow Lite documentation](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world)

## Running the scripts

First, having `conda` installed on your machine, you must create and activate the environment using:

```
conda env create --file environment_helloworld.yaml
conda activate tfl_helloworld
```
Then, the files in this folder are used as follows:

You must first run `tflite_helloworld_training.py` in order to train and save `sine_model_perfect.h5`, a fully connected neural network that approximates the sine function.


Then, to convert and quantize the model to a `.tflite` file, you can run `tflite_helloworld_converting.py` which will generate `sine_model_perfect_32float.tflite` and `sine_model_perfect_int8_justweights.tflite`.

For testing the `.h5` model within your PC you may run `tflite_helloworld_evaluating.py`, but if you want to evaluate the models implemented in the STM32 you must run `tflite_helloworld_evaluating_UART.py`.

### Running UART evaluation

The `tflite_helloworld_evaluating_UART.py` script is made to test *n* models implemented on the STM32 via UART. 

Afer running the file, the script will ask you how many times do you want the program to send *N* scalars between
0 and 2π, then it will ask you if you want it to send the following *N* numbers, just so you make sure the STM32 is connected and the desired `.tflite` model is loaded.

After accepting it will ask you for the name you will give to these *N* data points for the plot that will be made in the end.

You may change the implemented model in the microcontroller between each iteration.

After *n* models have been tested, the program will ask you if you want to add a *reference*, which will just be the actual sine of the inputs as calculated in your pc. 

Later the program will ask you if you want to save the data. This will be saved in `datos.pkl`. 

If you already have a `datos.pkl` file, you might want to run the program *n=0* times, not save them and instead load them from the `.pkl` file, which is the next question. 

> Future versions of this script will make it so that you can either choose to send *N* floats *n* times and save them or load the `.pkl` file without having to choose *n=0*.

In the end, this file produces a plot with all the tested models, using the defined legends yor each of them, and also calculates the mean absolute error of each of them.

> For each question, the pre-selected answer is written within brackets, so if the questions ends with `y/[n]`, unless you write either `n` or `N`, the `Yes` option will be selected.