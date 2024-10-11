# Alphabet Soup Charity Neural Network Analysis
## Project Overview
The purpose of this project is to create a binary classification model that predicts whether an organization funded by the Alphabet Soup charity will be successful. The original dataset consists of 34,299 organizations with metadata about each organization, such as:
  
  EIN and NAME—Identification columns
  APPLICATION_TYPE—Alphabet Soup application type
  AFFILIATION—Affiliated sector of industry
  CLASSIFICATION—Government organization classification
  USE_CASE—Use case for funding
  ORGANIZATION—Organization type
  STATUS—Active status
  INCOME_AMT—Income classification
  SPECIAL_CONSIDERATIONS—Special considerations for application
  ASK_AMT—Funding amount requested
  IS_SUCCESSFUL—Was the money used effectively

The analysis includes preprocessing the data, building and optimizing a neural network, and testing different techniques improve the model's accuracy.

## Instructions
### Step 1. Data Preprocessing
The following preprocessing steps were applied to the dataset:
1. Read in the charity_data.csv and identify the target variable(s) and feature variables for the model
  a. The target variable is IS_SUCCESSFUL, which indicates whether the organization was successful (1) or not (0).
  b.  Categorical features were converted into numerical values using one-hot encoding, including columns like APPLICATION_TYPE, AFFILIATION, and INCOME_AMT.
2. Drop uncessary columns: EIN and NAME
3. Identify the number of unique variables for each colum and for columns with more than 10 unique values (APPLICATION_TYPE and CLASSIFICATION) combine "rare" categorical varibales together into a new value, Other.
4. Encode categorical varibles using pd.get_dummies()
5. Split the preprocessed data into features array, and a target array
6. Use train_test_split to split the data into training and testing datasets
7. Use a StandardScalerScale instance and scale, train, and transform the datasets  

Binning: The ASK_AMT column, which represents the amount of funding requested by the organization, was binned into discrete categories. The bins were created to categorize the requests from less than $1K to more than $5M. These bins helped categorize and simplify the range of funding requests.

Bin Ranges:

<1K, 1K-5K, 5K-10K, 10K-50K, 50K-100K, 100K-500K, 500K-1M, 1M-5M, >5M
This approach was intended to help the model understand the relationship between the amount requested and the likelihood of success.

Outlier Removal: Outliers were identified using the interquartile range (IQR) method. The ASK_AMT and STATUS columns exhibited outliers, which were subsequently removed to prevent their influence on the model.

### 2. Compile, Train and Evaluate the Model 
The initial model was developed using TensorFlow's Keras Sequential API. The architecture consisted of the following layers:
  Input Layer: Based on the number of input features from the preprocessed data.
  Hidden Layers: Two hidden layers with 80 and 30 nodes, respectively, using the relu activation function.
  Output Layer: A single neuron with a sigmoid activation function to predict the binary outcome (IS_SUCCESSFUL).
  
  ### Training Results:
  Training Accuracy: 74.32%
  Test Accuracy: 73.13%
  The initial model produced a test accuracy of around 73%, which was below the target of 75%. So, further optimizations were attempted.

### 3. Optimization Attempts

#### Attempt 1: 
In the first optimization attempt, the model was adjusted to use:
100 nodes in the first hidden layer, 50 in the second, and 25 in the third.
![Optimizatin_1](<img width="1519" alt="Optimization_1 1" src="https://github.com/user-attachments/assets/dbad4fd8-a5af-40bf-b799-0049219a8ba4">
)

##### Results:
Training Accuracy: 74.70%
Test Accuracy: 72.47%

This shows did not show improvement in test accuracy compared to the initial model. 

#### Attempt 2: 
For the second optimization attempt, the activation function was changed from Relu to Tanh for the hidden layers and the model was adjusted to use:
80 nodes in the first hidden layer, 40 in the second, and 20 in the third.

![Optimization_2](<img width="1494" alt="Optimization_2 1" src="https://github.com/user-attachments/assets/82bb4b42-76e4-43df-83a8-c7d362e9e78e">
)

##### Results:

Training Accuracy: 74.04%
Test Accuracy: 73.43%
Using Tanh provided a slight improvement in test accuracy but still fell short of the 75% goal.

#### Attempt 3:
For the third optimization attempt,the ELU activation function was tested in all hidden layers.

![Optimization_3](<img width="1493" alt="Optimization_3 1" src="https://github.com/user-attachments/assets/0b71ece0-9986-4942-93ad-ca20fbca5b2c">
)

##### Results:

Training Accuracy: 74.25%
Test Accuracy: 73.40%
The ELU activation function showed similar results to Tanh, but did not achieve accuracy above 75%.

#### Attempt 4: Reducing Complexity of Hidden Layers
In the fourth optimization attempt, the number of epochs was reduced to 20, the first hidden has 80 nodes, second layer has 30 nodes, and the third layer has 10 nodes. Relu was used for the activation. 
![Optimization_4]()

##### Results:

Training Accuracy: 73.83%
Test Accuracy: 73.99%
This simplification brought test accuracy slightly higher but still did not meet the 75% goal.

#### 4. Binning Attempts
Next, the ASK_AMT column was binned into different categories to simplify the range of funding requests and provide the model with more interpretable categories. The following binning strategy was used:

<1K, 1K-5K, 5K-10K, 10K-50K, 50K-100K, 100K-500K, 500K-1M, 1M-5M, >5M
Binning + ReLU Model Results:
Training Accuracy: 73.89%
Test Accuracy: 73.51%
Binning with the ReLU model resulted in a slight improvement in test accuracy.

##### Binning + Tanh Model Results:
Training Accuracy: 74.47%
Test Accuracy: 72.61%
Binning with the Tanh model showed no significant improvement and was similar to previous attempts.

#### 5. Outlier Removal
Outliers were detected and removed from the dataset based on the ASK_AMT column. After removing the outliers, the model was retrained.

##### Results:
Training Accuracy: 76.13%
Test Accuracy: 71.16%
Even though the training accuracy improved after removing outliers, the test accuracy dropped slightly, indicating that removing outliers did not enhance the generalization of the model.

## Conclusion
Through multiple optimization attempts, the highest test accuracy achieved was 73.99%, which is still below the target of 75%. While adjustments in hidden layers, activation functions, binning, and outlier removal yielded improvements, the model still faces overfitting challenges. Further techniques, such as regularization, dropout, or alternative architectures (e.g., different types of neural networks), could be explored to improve the model's performance.
