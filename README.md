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
Code requires Python 2.7 and Theano 0.7. You can go to the [Setting Up page](https://github.com/flipkart-incubator/optimus/wiki/Setting-Up), for instructions on how to quickly set up the python environment required for this project. Requirements are also listed in the requirements.txt file.

## Start Using it
Visit the [Quick Start](https://github.com/flipkart-incubator/optimus/wiki/Quick-Start) guide to get started on using Optimus!

You can compare models trained using optimus to statistical models by using https://github.com/japerk/nltk-trainer, an awesome tool for easily training statistical classifiers. If you get some good results on a dataset, I would love to know about them! 

In case you face any issue, you can create an issue on github or send me a mail @ devashish.shankar@flipkart.com. Suggestions and improvements are also most welcome. I am setting up a contributors guide, which will have a code walkthrough if you want to contribute. Open github issues are a good place to start. 

## Core contributors
* Devashish Shankar ([@devashishshankar](https://github.com/devashishshankar))
* Prerana Singhal 
