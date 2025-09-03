# Description
A reimplementation of Repeated Elastic Net Technique (RENT) feature selection focused on improving compatibility with scikit-learn sister-libraries

Main differences from the [original RENT library](https://github.com/NMBU-Data-Science/RENT) are:
- Ability to specify custom model for calculating feature coefficients
- Ability to specify custom resampler for model training 
- Ability to specify custom feature selection criteria for evaluating feature coefficients

# Requirements
- numpy==2.2.5
- pandas==2.2.3
- scikit-learn==1.6.1
- scipy==1.15.2

# Reference
Jenul et al., (2021). RENT: A Python Package for Repeated Elastic Net 
Feature Selection. Journal of Open Source Software, 6(63), 3323, https://doi.org/10.21105/joss.03323
