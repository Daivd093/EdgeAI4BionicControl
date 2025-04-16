# Neural Network Models - Hello World Example

This folder contains the code used to generate the neural networks for the **Hello World** example. All models were developed in **Jupyter Notebooks**, and three different networks were created:

- A **fully connected neural network** that approximates the sine function. This model follows the *Hello World* example from the [TensorFlow Lite documentation](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) and this [DigiKey tutorial](https://www.youtube.com/watch?v=BzzqYNYOcWc).
- A **fully connected neural network** trained to classify the **Fashion MNIST** dataset. This serves as a baseline for image classification, where identifying correlations across channels is essential.
- A **convolutional neural network (CNN)** trained on the **Fashion MNIST** dataset. CNNs are commonly used in research for brainwave classification due to their spatial pattern recognition capabilities.

> *Ultimately, the final project used five fully connected networks, preceded by a data pre-processing stage. While more advanced architectures like CNNs or even RNNs stacked after CNN layers could have been explored, a simpler approach was adopted.*

> *The neural networks used in the final project are available in the `Dataset_and_Models` directory within the root folder.*

Originally, the STM32 deployment was intended to use **TensorFlow Lite Micro**, rather than **STM32Cube.AI**. Therefore, the models are saved both as `.tflite` files and as C header files (`.h`) for compatibility with embedded deployment.

To visualize any of these network architectures, you can use:
- [**Netron**](https://netron.app/) — a browser-based neural network visualizer.
- The **Model Analyzer** tool available within **STM32CubeIDE**.