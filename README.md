# Optimus
Quickly train, evaluate and deploy an optimum classifier for your text classification task. Currently, it allows you to train a CNN (Convolutional Neural Network) based text classifier. Using this toolkit, you should be able to train a classifier for most of the text classification tasks without writing a single piece of code. 

The main features of Optimus are:
* Easily train a CNN  classifier
* Config driven to make hyperparameter tuning and experimentation easy
* Debug mode: which allows you to visualize what is happening in the internal layers of the model
* Flask server for querying the trained model through an API

This project is based on: https://github.com/yoonkim/CNN_sentence (Many thanks to Yoon for open sourcing his code for the paper: http://arxiv.org/abs/1408.5882, which is arguably the best generic Deep Learning based text classifier at this point)
The improvements over the original code are:
* Multi-channel mode
* Complete refactoring to make the code modular
* Config driven, for easy experimentation
* Model serialization/deserialization
* Detailed evaluation results
* Model deployment on a Flask server
* Multi Class classification [In progress]
* Debug Mode [In progress]

This project is also inspired by https://github.com/japerk/nltk-trainer, which allows users to easily train NLTK based statistical classifiers. 

## Requirements
Code requires Python 2.7 and Theano 0.7. You can go to the Setting Up page, for instructions on how to quickly set up the python environment required for this project. Requirements are also listed in the requirements.txt file.

## Quick Start
### Train

`python train.py 	<model config file path> <training data file path> <file path to store classifier model> <true/false(preprocessing flag)>`

To train a toy model on a sample dataset run:

`python train.py sample/configs/SampleStaticConfig.json sample/datasets/sst_small_sample.csv  sample/myFirstModel.p true`

If this doesn't work, please go the Setting Up page, to ensure you have the dependencies installed.
We've placed the configs that worked best for us on a variety of tasks in the directory sample/configs. Feel free to experiment with them. Click here for complete user guide.

### Test

`python testing.py <model file path> <testing file path> <folder to store detailed output analysis> <preprocess? (true/false)> <load word vectors? (true/false)>`

To test the model you trained above, you can run:

`python test.py sample/myFirstModel.p sample/datasets/sst_small_sample.csv sample/outputNonStatic true false`

## Core contributors
* Devashish Shankar ([@devashishshankar](https://github.com/devashishshankar))
* Prerana Singhal 
