# AETS
=======
Source codes for Predicting Remaining Execution Time of Business Process In-stances via Auto-encoded Transition System

Requirement:
=======
Python 3.6  
pytorch 1.6.0

Code Structure:
=======
train.py: training function

/model:model files (MLP and autoencoder)

/aets/: different state representation functions are used to predict the remaining time (sequence,set and multiset)

/utils/data_pre_process.py: this file is used to construct the remaining time of the trace prefix.

/utils/construct_transition_system.py: construct trainsition system

/utils/metrics.py: MAE and MSE

