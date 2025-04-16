# In this folder we have:
- The RaimeSenAlgorithm.py, an implementation based of the work by [raimasen1729](https://github.com/raimasen1729/Finger-Flexion-Detection-Using-ECoG-Signal/tree/main
) to detect finger flexion using one f
- The conda environment needed to run the algorithm
- A model folder with the models for the NNs that predicts the flexion of each of the five fingers of the subject 1.
- A script to check the performance of those models

# In order to replicate what we have in this folder you must:

1. Download the [Dataset4 of the BCI C IV](https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip) and the [True Labels](https://www.bbci.de/competition/iv/results/ds4/true_labels.zip) and extract the files into the Dataset4 BCICIV folder

2. Activate conda environment

```
conda env create -f environment.yaml
conda activate ECoG
```

3. 