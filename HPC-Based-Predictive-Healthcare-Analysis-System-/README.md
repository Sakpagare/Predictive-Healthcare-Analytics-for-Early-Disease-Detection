# HPC-Based-Predictive-Healthcare-Analysis-System-
This project develops a predictive healthcare analytics system using machine learning and HPC for early disease detection. We implement a Random Forest model, comparing sequential vs. parallel execution (MPI) to optimize performance. The goal is to improve prediction speed and accuracy, enabling efficient real-time medical diagnosis.

## Overview

The HPC-Based Predictive Healthcare Analysis System is a project aimed at developing a predictive analytics system for early disease detection using machine learning and high-performance computing (HPC). This system leverages a Random Forest model to analyze medical records and improve prediction speed and accuracy, enabling efficient real-time medical diagnosis.

## Objectives

- Develop a predictive healthcare analytics system using machine learning.

- Implement a Random Forest model for disease prediction.

- Compare sequential vs. parallel execution (using MPI) to optimize performance.


## Features

- **Data Preprocessing:** Clean and preprocess medical records for analysis.

- **Model Training:** Train a Random Forest model on the dataset.

- **Noise Addition:** Expand the dataset with added noise to improve model robustness.

- **Performance Comparison:** Evaluate the performance of sequential vs. parallel execution.
  

## Performance Improvement

- **Sequential Execution Time:** 2.4001 seconds  
- **Parallel Execution Time:** 0.499 seconds  

### Speed-Up Formula  
The speed-up is calculated using the following formula:

- **Speed-up Percentage** = ((serial_time_new - parallel_time_new) / serial_time_new) * 100

- **Speed-Up:** 79.2% faster  

This demonstrates the performance acceleration achieved by parallelizing the code, reducing execution time by 79.2%.

## Accuracy Comparison

- **Sequential Code Accuracy:** 0.9119  
- **Parallel Code Accuracy:** 1.000  

### Accuracy Formula  
Accuracy is calculated using the following formula:
- **Accuracy** = ( Number of Correct Predictions /  Total Number of Predictions ) * 100

- **Accuracy Improvement:** 9.6% increase  

The parallel code achieves a perfect accuracy of 1.000, which is a 9.6% improvement over the sequential code's accuracy of 0.9119.


## License


This project is licensed under the Apache License 2.0. You may obtain a copy of the License at:


http://www.apache.org/licenses/LICENSE-2.0



## Acknowledgments


- **OpenMPI**: For providing a robust implementation of the Message Passing Interface (MPI) that enabled efficient parallel processing in our project.
- **Scikit-learn**: For the powerful machine learning library that facilitated the implementation of the Random Forest model.
- **Pandas**: For the invaluable data manipulation capabilities that were essential for data preprocessing.
- **NumPy**: For providing support for numerical computations, which were crucial for our analysis.
- **Matplotlib** and **Seaborn**: For their excellent data visualization libraries that helped in presenting our results effectively.
- **Jupyter Notebook**: For providing an interactive environment that made it easier to develop and document our code.

We also want to thank our peers and mentors for their feedback and support throughout the development of this project. Special thanks to Mr. Nileshchandra Pikle sir and Mrs. Srujna B. mam for their guidance and encouragement.
