# Performance Analysis of Neural Collaborative Filtering (NCF) Architecture

## Infrastructure for Advanced Analytics and Machine Learning, Sommersemester 2020

### Authors:
  * Bagrat Ter-Akopyan
  * Michał Filipiuk
  
Full description of the project can be found here: <link to our pdf file>  

The code used here was copied from [NVIDIA repository](https://github.com/NVIDIA/DeepLearningExamples/tree/17bc6aac816cbada40e799b06735c309f9b7043a/PyTorch/Recommendation/NCF).


## Quick Start Guide

1. Clone the repository.
```bash
git clone https://github.com/mkfilipiuk/AAML_Bagrat_Ter-Akopyan_Michal_Filipiuk
cd AAML_Bagrat_Ter-Akopyan_Michal_Filipiuk
```

2. In case of using a Linux machine you can create an Anaconda virtual environment from the environment.yml:
```bash
conda env create -f environment.yml
```
Otherwise:
```bash
conda create -n iaaml_ncf python=3.6
```
and install missing dependencies manually from the environment.yml

3. Activate the Anaconda virtual environment:
```bash
conda activate iaaml_ncf
```

4. Download and preprocess the data.

Preprocessing consists of downloading the data, filtering out users that have less than 20 ratings (by default), sorting the data and dropping the duplicates.
The preprocessed train and test data is then saved in PyTorch binary format to be loaded just before training.

No data augmentation techniques are used.

Download the data from https://grouplens.org/datasets/movielens/20m/ and put it into the ./data directory.

To preprocess the ML-20m dataset you can run:

```bash
./prepare_dataset.sh
```

Note: This command will return immediately without downloading anything if the data is already present in the `./data` directory.

This will store the preprocessed training and evaluation data in the `./data` directory so that it can be later
used to train the model (by passing the appropriate `--data` argument to the `ncf.py` script).

5. Start mlflow
```bash
mlflow server
```

6. Start training.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env ncf.py --data ./data/cache/ml-20m --checkpoint_dir ./data/checkpoints/
```

This will result in a checkpoint file being written to `/data/checkpoints/model.pth`.

7. To run the whole scripts for reproducing the complete results:
```bash
jupyter notebook
```
open the `training.ipynb` and run the cells

8. To reproduce the plots:
- open the `create_plots.ipynb`
- run the cells
