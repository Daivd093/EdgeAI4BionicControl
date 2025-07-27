# EdgeAI4BionicControl
Implementation of tinyML on STMicroelectronics' NUCLEO-L476RG to predict finger positions based on neural signals from Dataset 4 of BCI Competition IV.

To visualize any of the networks architectures, you can use:
- [**Netron**](https://netron.app/) — a browser-based neural network visualizer.
- The **Model Analyzer** tool available within **STM32CubeIDE**.

## Datasets_and_Models
This folder contains the code of the neural networks that are used to predict the fingers' positions.

## HelloWorld_Models
This folder contains the code of the neural networks that were used for a HelloWorld Project predicting the sine of the input. 

This code is largely based upon TensorFlow Lite's documentation.
## ST_Workspace
This folder contains all the project files from the STM32CubeIDE for both the hello world project and the finger flexion project, excluding the .metadata directory, as well as the build and debug output files.

<!-- ## Extra files

También tenemos el informe y la última ppt en español, tal vez en versiones posteriores lo traduzca manualmente o con IA incluso

## Future Work

- Traducir todo al inglés
- Modificar las cosas que dije en la ppt que me gustaría mejorar en el futuro

 -->

<!-- Todavía falta limpiar bastante el código -->