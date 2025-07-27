# Datates and Models for brain signal classification
This folder contains scripts to train, quantize and test neural networks trained for the Dataset 4 of the BCI Competition IV, a dataset of ECoG signals where the objective is to predict each of the fingers' positions as detected by a special glove. The full explanaiton of the Dataset can be find within the `Dataset4_BCICIV\` folder.

Perhaps the main file within this folder is `RaimeSenAlgorithm.py`, an implementation based on the work by [raimasen1729](https://github.com/raimasen1729/Finger-Flexion-Detection-Using-ECoG-Signal/tree/main
) to detect finger flexion using one fully connected neural network per finger.Although both the neural network and the training implementation, as well as the overall code structure have been changed, the preprocessing of the dataset is virtually the same, with some minor changes. The changes are mostly explained within the `.pdf` file on the root folder.

> Although in the current version that file is only written in spanish.

## Replicating the results:

1. Download the [Dataset4 of the BCI C IV](https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip) and the [True Labels](https://www.bbci.de/competition/iv/results/ds4/true_labels.zip) and extract the files into the Dataset4 BCICIV folder

2. Create and activate the conda environment

```
conda env create -f environment.yaml
conda activate ECoG
```

3. *Optional* Preprocess the dataset using `dataset2input.py` and save the preprocessed dataset into `Dataset4_BCICIV\sub{subject}_processed.npz`. This will be useful as you will need the preprocessed dataset to send it through UART to the microcontroller.

4. Train and save the neural networks for one of the 3 subjects of the dataset by using the `RaimeSenAlgorithm.py`. This script will try to open a `.npz` file with the preprocessed data before training, if it can't find it, it will do the preprocessing before training begins.

5. Either do conversion from `.h5` to `.tflite` using `conversion.py` or do a `int8` quantization using `quantization.py`. 
>Future versions will most likely have this two scripts merged.

6. The `testing.py` script will test the outputs of the neural networks based solely on the `.mat` with the outputs and the labels, it does not make inference.

7. Using the `testing_stm32.py` requires a STM32 with the `fingerflexiondetection` project running and the `VERBOSITY` macro set to `0`.

> The current version is fully sequential, and there is a massive bottleneck in the data transmission, so it takes between 2 and 3 hours to do the full testing in the microcontroller, while each inference takes just a few miliseconds.
