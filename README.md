# Alphabet Soup Charity Analysis
## Project Overview

The purpose of this project is to create a binary classification model that predicts whether an organization funded by the Alphabet Soup charity will be successful. The original dataset consists of 34,299 organizations with metadata about each organization, such as:
- EIN and NAME: Identification columns
- APPLICATION_TYPE: Alphabet Soup application type
- AFFILIATION: Affiliated sector of industry
- CLASSIFICATION: Government organization classification
- USE_CASE: Use case for funding
- ORGANIZATION: Organization type
- STATUS: Active status
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Special considerations for application
- ASK_AMT: Funding amount requested
- IS_SUCCESSFUL: Was the money used effectively

The analysis includes preprocessing the data, building and optimizing a neural network, and testing different techniques improve the model's accuracy.

## Instructions
### Step 1. Data Preprocessing

The following preprocessing steps were applied to the dataset:
1. Read in the charity_data.csv and identify the target variable(s) and feature variables for the model
  - The target variable is IS_SUCCESSFUL, which indicates whether the organization was successful (1) or not (0).
  - Categorical features were converted into numerical values using one-hot encoding, including columns like APPLICATION_TYPE, AFFILIATION, and INCOME_AMT.
2. Drop uncessary columns: EIN and NAME
3. Identify the number of unique variables for each colum and for columns with more than 10 unique values (APPLICATION_TYPE and CLASSIFICATION) combine "rare" categorical varibales together into a new value, Other.
    - For APPLICATION_TYPE, a cutoff value of 100 was used 
    - For CLASSIFICAtion, a cutoff value of 200 sas used
4. Encode categorical varibles using pd.get_dummies()
5. Split the preprocessed data into features array, and a target array
6. Use train_test_split to split the data into training and testing datasets
7. Use a StandardScalerScale instance and scale, train, and transform the datasets  

### 2. Compile, Train and Evaluate the Model 

The initial model was developed using TensorFlow's Keras Sequential API and consisted of the following layers:
  - Input Layer: Based on the number of input features from the preprocessed data.
  - Hidden Layers: Two hidden layers with 80 and 30 nodes, respectively, using the relu activation function.
  - Output Layer: A single neuron with a sigmoid activation function to predict the binary outcome (IS_SUCCESSFUL).
  
  ### Training Results:
  - Training Accuracy: 74.17%
  - Test Accuracy: 72.50%
  
  The initial model produced a test accuracy of around 73%, which was below the target of 75%. So, further optimizations were attempted.

### 3. Optimization Attempts

#### Attempt 1: 

In the first optimization attempt, the model was adjusted to use:
100 nodes in the first hidden layer, 50 in the second, and 25 in the third.

<img width="1519" alt="Optimization_1 1" src="https://github.com/user-attachments/assets/25ddac6a-e24d-4f49-918d-d8c6d70d41ae">

##### Results:
- Training Accuracy: 74.18%
- Test Accuracy: 73.53%

This shows a small improvement in test accuracy compared to the initial model. 

#### Attempt 2: 

For the second optimization attempt, the activation function was changed from Relu to Tanh for the hidden layers and the model was adjusted to use: 80 nodes in the first hidden layer, 40 in the second, and 20 in the third.

<img width="1494" alt="Optimization_2 1" src="https://github.com/user-attachments/assets/8bd595c3-e233-4533-9510-3a37390b3908">

##### Results:
- Training Accuracy: 74.78%
- Test Accuracy: 71.53%

Using Tanh provided a slight improvement in training accuracy but a decreased test accuracy. 

#### Attempt 3:

For the third optimization attempt,the ELU activation function was tested in all hidden layers.

<img width="1493" alt="Optimization_3 1" src="https://github.com/user-attachments/assets/859a7a3e-da8e-4ead-9b78-f6d555070fa8">


##### Results:

- Training Accuracy: 74.07%
- Test Accuracy: 73.06%

The ELU activation function showed similar results to the original model, but did not achieve accuracy above 75%.

#### Attempt 4: 

In the fourth optimization attempt, the number of epochs was reduced to 20, the first hidden layer had 80 nodes, second layer had 30 nodes, and the third layer had 10 nodes. Relu was used for the activation. 

![Optimization_4 1](https://github.com/user-attachments/assets/3201511e-b2a0-4a3c-aebe-32afe1f0827e)

##### Results:
- Training Accuracy: 74.13%
- Test Accuracy: 72.46%

This attempt did not show any improvement from the original model. 

#### Binning Attempts

Next, the ASK_AMT column was binned into different categories to simplify the range of funding requests. The following binning strategy was used: <1K, 1K-5K, 5K-10K, 10K-50K, 50K-100K, 100K-500K, 500K-1M, 1M-5M, >5M

<img width="1481" alt="Binning_Relu_1" src="https://github.com/user-attachments/assets/8ad20f9a-152e-4ef4-bec2-1bf0a6311334">


##### Binning + ReLU Model Results:
- Training Accuracy: 74.51%
- Test Accuracy: 72.56%

Binning with the ReLU model resulted in a slight improvement in test accuracy.

##### Binning + Tanh Model Results:
- Training Accuracy: 74.95%
- Test Accuracy: 72.38%

Binning with the Tanh model showed a did not show a greater improvement to the relu attempt. 

#### Outlier Removal
Lastly, an outlier plot was created to identify and remove outliers from the dataset based on the ASK_AMT column. 

<img width="1519" alt="Outlier1" src="https://github.com/user-attachments/assets/487f8fd9-dd35-4fc8-8fd1-12b16a64995b">
<img width="1532" alt="Outlier2" src="https://github.com/user-attachments/assets/b6888dd5-bd88-4cac-836a-7a6bcdf46a09">


##### Results:
- Training Accuracy: 75.71%
- Test Accuracy: 70.29%

Even though the training accuracy improved after removing outliers, the test accuracy dropped, suggesting that removing outliers did not improve the accuracy of the model.

## Conclusion

Ultimately,the first optimization attempt where the model was adjusted to use:
100 nodes in the first hidden layer, 50 in the second, and 25 in the third achieved the highest test accuracy at 73.53%. While adjustments in hidden layers, activation functions, binning, and outlier removal yielded improvements, the model still did not achieve an accuracy above 75%. Further techniques or the use of different types of neural networks, could be explored to improve the model's performance.

## Acknowledgements
    
    Xpert Learning Assistant was used to answer detailed questions, and assist in debugging.The starter code provided was the base of the report and was modified using course curriculum and activities to fit the requirements of the assignment. The TA and instructor for the course also assisted in adjusting the code during office hours.For more information about the Xpert Learning Assistant, visit [EdX Xpert Learning Assistant](https://www.edx.org/). 

## References
- [Activation Functions in Neural Networks](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
- [Automated Hyperparameter Tuning with Keras Tuner and TensorFlow 2.0](https://medium.com/analytics-vidhya/automated-hyperparameter-tuning-with-keras-tuner-and-tensorflow-2-0-31ec83f08a62#:~:text=A%20Hyperband%20tuner%20is%20an%20optimized%20version%20of,achieving%20the%20highest%20accuracy%20on%20the%20validation%20set.)

