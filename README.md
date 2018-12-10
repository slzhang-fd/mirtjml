[![Travis-CI Build Status](https://travis-ci.com/mrsta235/mirtjml.svg?branch=master)](https://travis-ci.com/mrsta235/mirtjml)

# mirtjml

Joint Maximum Likelihood Estimation for High-Dimensional Item Factor Analysis

## Description

The mirtjml package provides constrained joint maximum likelihood estimation
algorithms for item factor analysis (IFA) based on multidimensional item response theory
models. So far, we provide functions for exploratory and confirmatory IFA based on the 
multidimensional two parameter logistic (M2PL) model for binary response data. Comparing 
with traditional estimation methods for IFA, the methods implemented in this package scale
better to data with large numbers of respondents, items, and latent factors. The computation
is facilitated by multiprocessing 'OpenMP' API.
