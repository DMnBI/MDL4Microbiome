# MDL4Microbiome
MDL4Microbiome is a python source code for "MDL4Microbiome: Multimodal Deep Learning applied to classify healthy and disease states of human Microbiome"

A brief architecture of multimodal deep learning model.
------------------
A multimodal deep learning model aims for combining features from different modalities. Each feature generated by different methods is first fed to the classifier. The nodes of the last hidden layer are considered as embedded representations of each feature. Embedded representations are concatenated into a new shared repre-sentation inheriting original features. Combined feature representation is fed to the classifier for final classification.

<img src="https://user-images.githubusercontent.com/31638192/119304847-da697400-bca2-11eb-873a-a0de2de67da2.png" width="500">

Pre-installed python packages for MDL4Microbiome
------------------
MDL4Microbiome is implemented on Python version 3.6.9.

Following packages needs to be installed. The written versions are recommended; more recent versions are likely to work just fine. 

- numpy (version 1.16.6, https://numpy.org/install/)
- pandas (version 1.1.5, https://pandas.pydata.org/getting_started.html)
- keras (version 2.3.1, https://keras.io/about/#installation-amp-compatibility)
- sklearn (version 0.22.2, https://scikit-learn.org/stable/install.html)
- csv, argparse


Quick start guide
------------------
In the directory named `examples`, there are three sub-directories containing example datasets (i.e. IBD, T2D, LC datasets) and few files for command line arguments.

- Each dataset has three `.csv` files generated by individual features, i.e. conventional taxonomic profiles, genome-level relative abundance, and metabolic functional characteristics.

- Files that ends with `_datasets.txt` is a TXT file providing lists of files the user wants to feed into MDL4Microbiome. 

- Files that ends with `_ylab.txt` is a class label file for corresponding datasets. 

##### COMMAND LINE: 
> ##### ./MDL4Microbiome.py -m examples/LC/LC_datasets.txt -y examples/LC/LC_ylabel.txt -t tmp/ -e1 30 -e2 10 -l examples/results/LC_summarise

argument options (all required):
- `-m, --modality`: A file containing dataset file names (absolute path or relative path from current directory). One file per line.
- `-y, --ylabel`: A file containing class label (binary) of data points. The leading class label in alphabetical order is considered as positive state.
- `-t, --tmp`: A directory for temporary files to be saved.
- `-e1, --epoch1`: Number of epochs when training individual features.
- `-e2, --epoch2`: Number of epochs when training shared representation features.
- `-l, --log`: A name of a file for summarised results.
- `-i, --individual`: A flag for "individuals only". Use when you want to see the classification results of individual features only.

### Running MDL4Microbiome using your own data
If you have your own pre-processed datasets containing multiple different features (modalities), you can easily use MDL4Microbiome.
There are some requirements with the datasets.
- Note that the number of samples should match.
- Each dataset (`.csv` file) should have header on the first column and no index column. 
- MDL4Microbiome only supports binary classification. (label file should have 2 kinds of classes)
