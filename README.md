# A Rectification-Based Approach for Distilling Boosted Trees into Decision Trees

## Supplementary Material

## Contents

This archive contains the following resources:

### **`sources`**
The source code is built on the basis of a Git clone of https://github.com/crillab/pyxai

- `sources/pyxai/pyxai/examples/Distillation/`: This directory contains the code related to the experiments done in the paper; the main file (main.py) is the one to be launched.
- The main files to execute for all datasets are `run_all_default_DT.py` for default parameters or `run_all_optimised_DT.py` for optimized parameters.

### **`dataset`**
The datasets converted and used in our experiments.

More precisely, for each dataset, you can find:

- `datasets/<dataset>.csv`: The converted dataset.
- `datasets/<dataset>.types`: A JSON file containing all the information about the features.

### **`logs`**
The outputs produced by the algorithm run in the experiments.

- `logs/default_configuration/`: Results obtained without hyperparameter tuning (no limit on the depths of the decision trees) (see Table 3 in the paper).
- `logs/optimized_configuration/`: Results obtained when the depths of the decision trees have been tuned (see Table 4 in the paper).
- `logs/display_box_plots.ipynb`: Jupyter notebook used to generate the boxplots (see Figure 4 and Figure 5 in the paper).

### **`proofs.pdf`**
The proofs of the propositions provided in the paper.

### **Figures**
- `Relative accuracy $I_P$ obtained after rectification or retraining for the dataset under the default configuration.pdf`: Figures of relative accuracy $I_P$ for each dataset under default configuration (Figure 4).
- `Relative accuracy $I_P$ obtained after rectification or retraining for the dataset under the optimized configuration.pdf`: Figures of relative accuracy $I_P$ for each dataset under optimized configuration (Figure 5).
- `time_unresolved_instances.pdf`: Figures of comparison of unresolved instances over time between P and I for each dataset (Figure 3).

### **`BA-Trees-master`**
The Born-Again Tree Ensembles (BATE) approach, used as a point of comparison in Section 6 of the paper.

Please follow the instructions for using Born-Again Tree Ensembles at https://github.com/vidalt/BA-Trees. Instead of cloning the software and installing the packages necessary to run Born-Again, please use the source provided in this archive.

- `BA-Trees-master/BA-Trees-master/src/resources/forests/DATA_SET`: This directory contains our datasets used with the BATE approach.
- `BA-Trees-master/BA-Trees-master/src/born_again_dp/logs`: This directory contains the logs related to the BATE experiments with our datasets.
- The main file to execute for all datasets is `script.sh`.
- `output_6.tree` and `output_7.tree`: These files contain the decision trees obtained for COMPAS and Contraceptive using BATE.

### **`xreason`**
Using MaxSAT for Efficient Explanations of Tree Ensembles, used as a point of comparison in Section 5 of the paper.

Please follow the instructions for using xreason at https://github.com/alexeyignatiev/xreason. Instead of cloning the software and installing the packages necessary to run XReason, please use the source provided in this archive.

- `xreason/aaai22/bench/ann-thyroid/`: This directory contains our datasets used with the MaxSAT approach.
- `xreason/src/results/`: This directory contains the logs related to the MaxSAT, SMT and Decision Tree experiments with our datasets (log of comparison time between MaxSAT, SMT and Decision Tree) see Table 5.
- `xreason/src/results/DT/Comparison_of_unresolved_instances_over_time`: Jupyter notebook used to generate Figure 3 in the paper and comparison of unresolved instances over time (time_unresolved_instances.pdf).
- The main file to train all datasets is `aaai22-train-all.py`.
- Given the trained models, the main file to run the experimentation script is `aaai22-experiment.py`.
- `xreason/src/temp/`: These files contain the BT obtained for datasets using xreason library.

## Setup

- Be sure to use a Linux OS and a version of Python >= 3.8.
- Install Pyxai. Follow these [instructions](https://www.cril.univ-artois.fr/pyxai/documentation/installation/github/). Instead of cloning the software, please use the source provided in this archive.
- Install the required dependencies:

```bash
python3 -m pip install numpy==2.0.2
python3 -m pip install pandas==2.2.3
python3 -m pip install scikit-learn==1.5.2
python3 -m pip install xgboost==1.7.3
```

- To compile the modified version of pyxai in the pyxai directory:

```bash
python3 -m pip install -e .
```

## How to Use Our Program

- For a given dataset, the program `sources/pyxai/pyxai/examples/Distillation/main.py` implements the two approaches presented in the paper: the retraining approach and the rectification approach.

- First, navigate to the directory:

```bash
cd sources/pyxai/pyxai/examples/Distillation/
```

- For the default configuration of the decision tree:

```bash
python3 main.py -dataset=../datasets/cleveland_nominal_2_0
```

- For the optimized configuration of the decision tree:

```bash
python3 main.py -dataset=../datasets/cleveland_nominal_2_0 -types=True
```

- The program returns 3 files that can be found in the same directory:
  - `train_test_data.csv`: Instances of the training set L (before binarization).
  - `validation_data.csv`: Instances of the validation set V (before binarization).
  - `dataset.json`: The log file containing the results (in the same form as the ones given in the `logs/` directory).

- **Note:** Running the code may yield different results for the retraining method due to the random selection of instances matching the classification rule that has been derived. The conclusions presented in the paper are not impacted by this variability.

  In contrast, the rectification approach is deterministic.