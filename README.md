# GPGPU-Particle-Filter
#Author: Rita Leite


This project consists of an implementation of a bootstrap particle filter on GPU. This project was done within the course GPGPU Computing in the Lappeenranta University of Technology.


##Instructions
To run the bootstrap filter algorithm simply execute the file matlab_script.m on matlab

##Files
  -log.cu: contains necessary CUDA functions to compute log-likelihoods and sample weights
  -pendulum.cu: contains necessary CUDA functions to perform state propagation of the pendulum model
  -matlab_script: script used for calling the CUDA kernels and performing bootstrap filter algorithm
  -script_for_matlab_version: contains a matlab implementation of bootstrap filter, used to benchmark runtime
