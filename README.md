# TwitterSentimentAnalysis_PCML_EPFL
# Group 3: Joel Castellon, Mathieu Clavel, Marta Kuziora

Dependencies
============
> Python 3.5.2
> tensorflow 0.12.0-rc0
> nltk 3.2.1
> sklearn 0.18.1

Before running any code please make sure all data files (either
for training or testing) you
are going to use are in /data/twitter-datasets/


Training
===========
The baselines have been trained using scikit-learn and
Tensorflow for the CNN.
For the process and plot generating comparisons between these different
methods see Baselines.ipynb

The main model we use (CNN) can be trained with cnn_cross.py
for all available options: cnn_cross.py --help

Example for training:
./cnn_cross.py --channels="glove, word2vec" --iters_per_fold=200 --hparams="0.2,0.005 0.5,0.05 0.7,0.1"

#Note:
./cnn_cross.py looks for glove_embeddings.npy and word2vec_embeddings.npy
in the home folder for the project.
Pre calculated embeddings available at :
https://www.dropbox.com/sh/k7v3x0ttni0bdzb/AABNREh1tZuRK8NV-Ym5SY_Pa?dl=0 

If some of these embedding files are not found the script will train
GloVe and Word2Vec models and generate those (can take very long for
train_[pos|neg]_full.txt). 

Evaluation
===========
./run.py --eval_train --checkpoint_dir="./runs/run-id/checkpoints/"

where the run-id folder is generated every train and for every
fold in cross-validation. 

Note: The dev error that is printed upon evaluation is just a 
reference we used for tunning the model. It's an augmented data set
used for validation where we generated some tweets with a language model (http://nlp.stanford.edu/sentiment/).
This is a common trick in deep learning.

Kaggle
===========
**IMPORTANT**
The best run for the Kaggle submission requires the run-id file available at:
https://www.dropbox.com/s/0legtswltjw36cr/1482161789.zip?dl=0

Please uncompress and place it at /runs/  folder (you should have /runs/1482161789/ folder). Then, the pre trained model will create a file runs/1482161789/predictions.csv (Kaggle submission).

By default, the pre-trained model looks for a file in /data/twitter-datasets/test_data.txt

If you want to generate predictions from the model on another file that is not
the one that was provided during the competition:
./run.py --eval_train --checkpoint_dir="./runs/1482161789/checkpoints/" --test_data_file=PATH_TO_NEW_TEST_FILE.

To reproduce the Kaggle results just run the above command without the --test_data_file flag.



