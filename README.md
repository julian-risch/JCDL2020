# Hierarchical Document Classification as a Sequence Generation Task

Collection of standard and sequence to sequence deep learning models for the task of Automatic Patent Classification:

- [Convolutional Neural Network for Sentence Classification [1] ](#references) 
- [Gated Recurrent Unit Network  [2] ](#references) 
- [Label-Embedding Attentive Model (LEAM) [3] ](#references) 
- [Show, Attend and Tell [4] ](#references) modified to be used with texts, called Read, Attend and Label 
- [Tree Convolutional Neural Network [5] (NOT PORTED YET) ](#references) 
- [Transformer / Transformer Encoder [6] ](#references)

All the implementations are based on Tensorflow. For backward compatibility all the code must run with TF v1.0. Originally the repository was developed with TF 1.13 but it should run just fine with TF 1.14 as well, although it is not suggested (lots of annoying warnings).

## Citation

If you use our work, please cite our paper [**Hierarchical Document Classification as a Sequence Generation Task**](https://github.com/julian-risch/JCDL2020/raw/master/risch2020hierarchical.pdf) as follows:

    @inproceedings{risch2020hierarchical,
    author = {Risch, Julian and Garda, Samuele and Krestel, Ralf},
    booktitle = {Proceedings of the Joint Conference on Digital Libraries (JCDL)},
    title = {Hierarchical Document Classification as a Sequence Generation Task},
    year = {2020}
    }

# Data

1) Download the Patent Classification Benchmark Dataset with 2 million patents [USPTO-2M](http://mleg.cse.sc.edu/DeepPatent/index.html). Please place all the files in a single folder.

2) Download the fasttext based [domain-specific word embeddings](https://hpi.de/naumann/projects/web-science/deep-learning-for-text/patent-classification.html)

# How To


Clone the repo

    git clone https://github.com/julian-risch/JCDL2020
    

## IPC 

Create parsed IPC file and mapping (IPC label to description). This step has been carried out already. You can find the necessary files in the folder `data/ipc`.

The files can be obtained by calling the following:

    python3 -m scripts.ipc --ipc <IPC.XML> --parsed <IPC.JSON> --mapping <IPC.PKL>
    
This script works with IPC version 2018. Hence if you wish to change the IPC scheme, e.g. a different year, there is no garuantee that this will work with other versions.

## Preprocess data and Create embeddings

Go through the dataset a first time to collect vocabulary statistics:

    python3 -m scripts.parse_uspto2m --dir <upsto2m original> --out <parsed_uspto2m>

The processed files are saved in a compressed format.

Go through the dataset a second time to create tfrecords file for a fast input pipeline with `tf.data.Dataset` module:
    
    python3 -m scripts.data2tfrecords --dir <parsed_uspto2m> --task uspto2m --out <tfrecords_uspto2m> [--max-freq 50] 

All necessary files will be saved in "tfrecords_uspto2m" folder. It is possible to limit the number of words to consider for encoding the patent abstracts with `--max-freq`.

Finally you can create the embeddings matrix necessay for the models with:
    
    python3 -m scripts.embeddings --path <fasttext.bin> --mapping <IPC.PKL> --out <tfrecords_uspto2m>
    
Plese be sure to pass the `--out` argument the folder containing the "tfrecord" files.

## Train

Now it is possible to train one of the available models. For this you need to use one of the configuration file that you can find in the `configs` folder. In the repository there are already present an example file for each of the available model. 

To train the model you can invoke the following:

    python3 train.py --config <config/model.json> --task uspto2m --results <res_dir> --models <models_dir> --jobs <cores>
    

By default a copy of the model at the end of each epoch will be saved in the folder passed to `--models`. In order to get an overview of the effect of the hyperparameters in the blink of an eye the validation accuracy of the last epoch will be saved in a file created into the folder `--results`. 

Be carefull that the file name containing the results will be created automatically given the name of the configuration file. Hence, in order to avoid overwriting you need to create a different configuration file for each hyperparameter setting. This is tedious but pays in tidiness.

## Predict

Once you trained the model you can use it to create a prediction file that will be used for the final evaluation on the test set.

    python3 predict.py --config <config/model.json> --task uspto2m --model <models_dir/model/> --out <pred_dir> --jobs <cores>
    
This will create a folder "pred_dir" where a file (name after the configuration file) containing the predictions obtained with the model will be saved. Be sure to pass to `--model` the path to the model folder.


# Requirements

- tensorflow == 1.13.0

- gensim == 3.7.3

- pandas == 0.23.4

# References

[1] Li, Shaobo, et al. "DeepPatent: patent classification with convolutional neural networks and word embedding." Scientometrics 117.2 (2018): 721-744.

[2] Risch, Julian, and Ralf Krestel. "Learning Patent Speak: Investigating Domain-Specific Word Embeddings." Proceedings of the Thirteenth International Conference on Digital Information Management (ICDIM 2018), Porto, Portugal. 2018.

[3] Wang, Guoyin, et al. "Joint embedding of words and labels for text classification." arXiv preprint arXiv:1805.04174 (2018).

[4] Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. 2015.

[5] Yan, Yan. "Hierarchical Classification with Convolutional Neural Networks for Biomedical Literature." International Journal of Computer Science and Software Engineering 5.4 (2016): 58.

[6] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.


