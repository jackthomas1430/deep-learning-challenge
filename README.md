# Alphabet Soup Charity Neural Network Analysis
## Project Overview
The goal of this project is to develop a binary classification model that predicts whether an organization funded by the Alphabet Soup charity will be successful. The original dataset consists of 34,299 organizations with various features such as the type of application, the amount requested, and several other categorical and numerical attributes. The analysis involves preprocessing the data, building and optimizing a neural network, and testing different techniques like binning and outlier removal to improve the model's accuracy.

### 1. Data Preprocessing
The following preprocessing steps were applied to the dataset:

Target Variable: The target variable is IS_SUCCESSFUL, which indicates whether the organization was successful (1) or not (0).

Features: Various categorical features were converted into numerical values using one-hot encoding, including columns like APPLICATION_TYPE, AFFILIATION, and INCOME_AMT.

Binning: The ASK_AMT column, which represents the amount of funding requested by the organization, was binned into discrete categories. The bins were created to categorize the requests from less than $1K to more than $5M. These bins helped categorize and simplify the range of funding requests.

Bin Ranges:

<1K, 1K-5K, 5K-10K, 10K-50K, 50K-100K, 100K-500K, 500K-1M, 1M-5M, >5M
This approach was intended to help the model understand the relationship between the amount requested and the likelihood of success.

Outlier Removal: Outliers were identified using the interquartile range (IQR) method. The ASK_AMT and STATUS columns exhibited outliers, which were subsequently removed to prevent their influence on the model.

### 2. Initial Model Development
The initial model was developed using TensorFlow's Keras Sequential API. The architecture consisted of the following layers:

Input Layer: Based on the number of input features from the preprocessed data.
Hidden Layers: Three hidden layers with 100, 50, and 25 nodes, respectively, using the ReLU activation function.
Output Layer: A single neuron with a sigmoid activation function to predict the binary outcome (IS_SUCCESSFUL).
Training Results:
Training Accuracy: 74.66%
Test Accuracy: 71.31%
The initial model produced a test accuracy of around 71%, which was below the target of 75%. Therefore, further optimizations were attempted.

### 3. Optimization Attempts
#### Attempt 1: Adjusting Hidden Layers and Node Counts
In the first optimization attempt, the model was adjusted to use:
100 nodes in the first hidden layer, 50 in the second, and 25 in the third.

##### Results:
Training Accuracy: 74.70%
Test Accuracy: 72.47%
This shows an improvement in test accuracy compared to the initial model, but it still falls short of the 75% goal.

#### Attempt 2: Tuning Activation Functions
For the second optimization attempt, the activation function was changed from ReLU to Tanh for the hidden layers and the model was adjusted to use:
80 nodes in the first hidden layer, 40 in the second, and 20 in the third.

##### Results:

Training Accuracy: 74.04%
Test Accuracy: 73.43%
Using Tanh provided a small improvement in test accuracy but still fell short of the 75% goal.

#### Attempt 3: Experimenting with ELU Activation
In the third optimization, the ELU activation function was tested in all hidden layers.

##### Results:

Training Accuracy: 74.25%
Test Accuracy: 73.39%
The ELU activation function showed similar results to Tanh, again falling short of the target.

#### Attempt 4: Reducing Complexity of Hidden Layers
In the final optimization attempt, the number of nodes was significantly reduced to simplify the model and prevent overfitting.

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
