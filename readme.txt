The directory contains the following Matlab functions:

example_code.m generates data for an XOR gate. Then trains a predictive coding network, as well as the equivalent MLP on the data.

f.m - calculates the an activation function.

f_b.m - calculates the an activation function as well as its derivitaive.

w_init.m - initialises a set of random weights, for a given network structure

(The following codes only accept one data point at a time)

test - makes a prediction for an ann/pc network + outputs rmse

rms_error - calculated rmse

learn_ann - performs back-propagation

learn_pc - performs the learning for a predictive coding network

infer_pc - performs the inference stage
