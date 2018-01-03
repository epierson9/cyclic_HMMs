# Cyclic Hidden Markov Models

This repo implements Cyclic Hidden Markov Models (CyHMMs), described in ["Modeling Individual Cyclic Variation in Human Behavior"](https://arxiv.org/abs/1712.05748). See illustration_of_how_to_run_model.ipynb for sample code and suggestions for using the CyHMMs. We unfortunately cannot provide the health datasets used in the original paper because they are not public, so we demonstrate the model on simulated data.  

This code requires pomegranate, scipy, numpy, sklearn, pandas, and matplotlib for full functionality. It was tested using Python 2.7.5, pomegranate 0.6.1, scipy 0.18.1, numpy 1.11.3, sklearn 0.18.1, pandas 0.19.2, and matplotlib 2.0.0. 

### File descriptions:

**illustration_of_how_to_run_model.ipynb**: illustrates how to run the model.  

**cyclic_HMM.py**: contains the implementation of CyHMMs. 

**test_on_simulated_data.py**: generates simulated data.

**constants_and_util.py**: miscellaneous helper methods and constants. 

**final_continuous_params.csv,final_binary_params.csv**: CSVs which contain saved model parameters; used for unit testing. 

### Citation:
If you use this work, please cite: 

Emma Pierson, Tim Althoff, and Jure Leskovec. "Modeling Individual Cyclic Variation in Human Behavior." WWW 2018.

### Contact: 
Contact Emma Pierson (emmap1@cs.stanford.edu) with any questions or suggestions!


