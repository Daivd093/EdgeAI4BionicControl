# ST Workspace
These folders contain all the relevant files to implement the "hello world" and "finger flexion" projects on STM32CubeIDE v1.18.1 using STM32Cube.AI v10.0.0.

To change the `.tflite` model implemented you must access the `.ioc` file, and then `Pinout & Configuration > Middleware & Software Packs > X-CUBE-AI > Configuration > [model name]` to chose the desired `.tflite` model.

Both projects hold a macro named `VERBOSE` that, if set as `1` will print some debugging messages via UART. In order to run the evaluating scripts from the other folders it is mandatory to have `#define VERBOSE 0`.

<!-- Podría revisar bien cómo se hace lo de implementar proyectos en STM32CubeIDE en base a estas carpetas y poner algo más acá después de borrar mis archivos locales -->

## nucleo-l476rg-edge-ai-helloworld-sine
This folder contains the STM32 project Hello World to test the capabilities of the hardware using the .tfile sine_models available in `EdgeAI4BionicControl\HelloWorld_Models\`.




## nucleo-l476rg-edge-ai-finger-flexion
This folder contains the final STM32 project that implements the FingerFlexionEstimating NeuralNerworks into a Nucleo64 L476RG.

<!-- No sé si comentar aquí o en otro lado lo secuencial que es este proyecto -->




<!-- Podría poner instrucciones para replicar el proyecto en otro STM32 -->