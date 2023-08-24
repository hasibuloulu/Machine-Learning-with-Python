# Machine-Learning-with-Python
Machine Learning with Python Exercise

1. Introduction OF Statistics (https://github.com/piyushpathak03/Machine-learning-algorithm-PDF)
- There Are Mainly Two types of data
	- Numerical			- Categorical
	
In Numerical it divided further two types:
      a. Discrete (Use in classification)  
      b. Continuous (Use in Regression)

a.  Discrete Data Type: The Value Which Counted as Whole or fixed value. ex. 1,45,73 but not 23.42,36.54. Integer value is 
    a discrete value. ex. age, no. of vehicle
    
b. Continuous Data Type: Here Value contains range, measurement. eg. 1.3,1.5,1.6 but not 1,5,6. it is floating type value. 
   Eg. weight in kg, House price

2. Categorical Also Divided into Two types:
      a. Ordinal      		 b. Nominal

a. Ordinal: These are meaningfully ordered. Eg. Rating of Product, Grade of Mark. Here the Quality Increases from Lower to 
   Higher and Higher to Lower in order.

b. Nominal: Here No Intrinsic order of the label. all labels are not in order. Eg. Color - as in color there are many 
   color like Red, green, black etc but can you find any order between them, Ans is no so these type are nominal. 
   Basically, Categorical are in String (Object) Format. Numerical is Float or int Format in the dataset.

Types Of Statistics. Mainly Two Types: 	1. Descriptive 		2. Inferential

  1. Descriptive Statistics: This is used to describe things, frequently group of peoples. Eg. Mean, Median, Mode, 
     Variance, Standard Deviation
     
  3. Inferential Statistics: This is used to make inference and draw conclusions. Eg. T-Tset, Chi-Square, Anova Test
     Variance is a measure that shows how far is each value from a particular point, preferably mean value. 
     Mathematically, it is defined as the average of squared differences from the mean value.

The probability Density Function (PDF) or density of a continuous random variable, is a function whose value at any given sample (or point) in the sample space (the set of possible values taken by the random variable) can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample. Types:
	- Binomial Distribution (It is use when there is more than one outcome of the certain experiment.)
	- Poisson Distribution (It is used to find probability in between some time period/ time interval)
	- Normal Distribution (Gaussian Distribution) [behaviours of most of the situations in the universe]
	- Bernoulli Distribution
	- Uniform Distribution
	- Exponential Distribution

Bias is the difference between Predicted Value and Actual Value.
   Low Bias: Predicting less assumption about Target Function (Predict Value >> Actual Value) - the difference is Very 
   Small Between predict and Actual Value
   
   High Bias: Predicting more assumptions about Target Function (Predict Value >>>>>> Actual Value) - the difference is 
   very much large between predict and Actual Value	
   Variance is the amount that the estimate of the target function will change if different training data is used. It 
   determines how spread of predicts value each other.
   
 - Low Variance: Predicting small changes to the estimate of the target function with changes to the training dataset.
 - High Variance: Predicting large changes to the estimate of the target function with changes to the training dataset.
   Underfit models usually have high bias and low variance. underfitting happens when a model is unable to capture the 
   underlying pattern of the data. It happens when we have very less amount of data to build an accurate model or when we 
   try to build a linear model with a nonlinear data.  Also, these kinds of models are very simple to capture the complex 
   patterns in data like Linear and logistic regression.
   
Overfitting models have low bias and high variance. Overfitting happens when our model captures the noise along with the underlying pattern in data. It happens when we train our model a lot over the noisy dataset.
Overfit: The model is overly trained to the dataset, which may capture noise and produce a non-generalized model. Using too many independent variables would lead over fit the model. ver-fitting if the number of features is much greater than the number of samples.

Cost Function: is the difference between the actual values of y and our model output y hat. Gradient descent is a technique to use the derivative of a cost function to change the parameter values, in order to minimize the cost function. 

What is the Bias-Variance Trade-off?
 - If our model is too simple and has very few parameters then it may have high bias and low variance.
 - On the other hand, if our model has a large number of parameters then it’s going to have high variance and low bias.
 - So, we need to find the right/good balance without overfitting and underfitting the data.
 - There is no escaping the relationship between bias and variance in machine learning. 
   *increasing the bias will decrease the variance. * Increasing the variance will decrease the bias.
 - If the algorithm is too simple then it may be on high bias and low variance condition and thus is error-prone. 
   If algorithms fit too complex, then it may be on high variance and low bias. In the latter condition, 
   the new entries will not perform well. Well, there is something between both of these conditions, known as Trade-off or
   Bias Variance Trade-off.
   
Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting, i.e., failing to generalize a pattern. The three steps involved in cross-validation are as follows:
		1. Reserve some portion of sample dataset.
		2. Using the rest dataset train the model.
		3. Test the model using the reserve portion of the dataset.		
  
Ensemble Learning methods combine several trees base algorithms to construct better predictive performance than a single tree base algorithm. The main principle behind the ensemble model is that a group of weak learners come together to form a strong learner, thus increasing the accuracy of the model. When we try to predict the target variable using any machine learning technique, the main causes of difference in actual and predicted values are noise, variance, and bias. Ensemble helps to reduce these factors (except noise, which is irreducible error).

Gradient Descent is an optimization algorithm used for minimizing the cost function in various machine learning algorithms. It is basically used for updating the parameters of the learning model. Types of gradient Descent:
Batch Gradient Descent: This is a type of gradient descent which processes all the training examples for each iteration of gradient descent. But if the number of training examples is large, then batch gradient descent is computationally very expensive. Hence if the number of training examples is large, then batch gradient descent is not preferred. Instead, we prefer to use stochastic gradient descent or mini-batch gradient descent.

Stochastic Gradient Descent: This is a type of gradient descent which processes 1 training example per iteration. Hence, the parameters are being updated even after one iteration in which only a single example has been processed. Hence this is quite faster than batch gradient descent. But again, when the number of training examples is large, even then it processes only one example which can be additional overhead for the system as the number of iterations will be quite large.
Mini Batch gradient descent: This is a type of gradient descent which works faster than both batch gradient descent and stochastic gradient descent. Here b examples where b<m are processed per iteration. So even if the number of training examples is large, it is processed in batches of b training examples in one go. Thus, it works for larger training examples and that too with lesser number of iterations.

Evaluation Matric:
Confusion Matrix: The Confusion matrix is one of the most intuitive and easiest metrics used for finding the correctness and accuracy of the model. It is only used for Classification problem where the output can be of two or more types of classes. There are 4 terms you should keep in mind:
    1. True Positives (TP): It is the case where we predicted Yes and the real output was also yes.
    2. True Negatives (TN): It is the case where we predicted No and the real output was also No.
    3. False Positives (FP): It is the case where we predicted Yes but it was actually No.
    4. False Negatives (FN): It is the case where we predicted No but it was actually Yes.
    
Accuracy: Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right.  Accuracy is a good measure when the target variable classes in the data are nearly balanced. Accuracy should NEVER be used as a measure when the target variable classes in the data are a majority of one class (Imbalanced dataset)

Precision: What proportion of positive identifications was actually correct?
Recall: What proportion of actual positives was identified correctly?
Sensitivity: is the metric that evaluates a model’s ability to predict true positives of each available category. Specificity is the metric that evaluates a model’s ability to predict true negatives of each available category.

Classification of Matrics 
F1-Score: F1-score is a metric which considers both, precision and recall as we can’t always evaluate both and then take the higher one for our model. It is the harmonic mean of precision and recall. It tells us about the balance that exists between precision and recall. This is the harmonic mean of Precision and Recall and gives a better measure of the incorrectly classified cases than the Accuracy Metric. F1-score is used when the False Negatives and False Positives are crucial. F1-score is a better metric when there are imbalanced classes. In most real-life classification problems, imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on. 

ROC And AUC Curve:
  TPR (True Positive Rate): the true positive rate, also referred to sensitivity or recall, is used to measure the         
  percentage of actual positives which are correctly identified.
  
  TNR (True Negative Rate): The Specificity of a test, also referred to as the true negative rate (TNR), is the proportion 
  of samples that test negative using the test in question that are genuinely negative.
  
  FPR (False Positive Rate): In statistics, when performing multiple comparisons, a false positive ratio (also known as 
  fall-out or false alarm ratio) is the probability of falsely rejecting the null hypothesis for a particular test. The 
  false positive rate is calculated as the ratio between the number of negative events wrongly categorized as positive 
  (false positives) and the total number of actual negative events (regardless of classification).
  
  FNR (False Negative Rate): The rate of occurrence of negative test results in those who have the attribute or disease 
  for which they are being tested.
  
ROC (relative operating characteristic) curve is one of the important evaluation metrics that should be used to check the performance of a classification model. It is a comparison of two main characteristics (TPR and FPR). It is plotted between sensitivity (recall) and False Positive Rate (FPR = 1-specificity). 

AUC is also called as AREA UNDER CURVE. It is used in classification analysis in order to determine which of the used models predicts the classes best. An example of its application is ROC curves. AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0 and if the predictions are 100% correct has an AUC of 1.

Regression Model Metric:
  1. Mean Squared Error (MSE) [it is simply the average of the squared difference between the target value and the value 
predicted by the regression model]
  2. Root-Mean-Squared-Error (RMSE) [It is the square root of the averaged squared difference between the target value and the value predicted by the model]
  3. Mean-Absolute-Error (MAE). [absolute difference between the target value and value predicted by the model]
  4. Root Mean Squared Logarithmic Error (RMSLE).
  5. R² or Coefficient of Determination. (The metric helps us to compare our current model with a constant baseline and tells us how much our model is better. The constant baseline is chosen by taking the mean of the data and drawing a line at the mean. R² is a scale-free score that implies it doesn't matter whether the values are too large or too small, the R² will always be less than or equal to 1)
  7. Adjusted R² (it is suitable for multiple linear regression where more than one independent variable)

Regularization is an extremely important concept in machine learning. It is a way to prevent overfitting, and thus, improve the likely generalization performance of a model by reducing the complexity of the final estimated model. In regularization, what we do is normally we keep the same number of features but reduce the magnitude of the coefficients. The Main Objective of regularization is to scale down the coefficient Value so that overfit not happen in model.

ML technique:
-	Regression (predicting continuous values)
-	Classification (predicting the items of class)
-	Clustering (finding structure of data)
-	Associations (associating frequent co-occurring items/events)
-	Anomaly detection (discovering abnormal and unusual cases)
-	Sequence mining (predicting next events, click stream (Markov model, HMM))
-	Dimension reduction (reducing the size of data (PCA)
-	Recommendation systems (recommending items)

Model evaluation Approaches:
-	Train and test on the same dataset
-	Train/test split (avoid out-of-sample accuracy)

Supervised learning, it is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross-validation process. can be separated into two types of problems when data mining—classification and regression.

    •	Classification uses an algorithm to accurately Supervised learning assign test data into specific categories. It 
      recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should 
      be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), 
      decision trees, k-nearest neighbor, and random forest, which are described in more detail below.
    •	Regression is used to understand the relationship between dependent and independent variables. It is commonly used 
      to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and 
      polynomial regression are popular regression algorithms.
      
Supervised learning algorithms

Neural networks: Primarily leveraged for deep learning algorithms, neural networks process training data by mimicking the interconnectivity of the human brain through layers of nodes. Each node is made up of inputs, weights, a bias (or threshold), and an output. If that output value exceeds a given threshold, its “fires” or activates the node, passing data to the next layer in the network. Neural networks learn this mapping function through supervised learning, adjusting based on the loss function through the process of gradient descent. When the cost function is at or near zero, we can be confident in the model’s accuracy to yield the correct answer.

Linear regression: Linear regression is used to identify the relationship between a dependent variable and one or more independent variables and is typically leveraged to make predictions about future outcomes. When there is only one independent variable and one dependent variable, it is known as simple linear regression. As the number of independent variables increases, it is referred to as multiple linear regression. For each type of linear regression, it seeks to plot a line of best fit, which is calculated through the method of least squares. However, unlike other regression models, this line is straight when plotted on a graph. Linear Model: Simple Linear Regression is finding the best relationship between the input variable x (independent variable) and the expected variable y (dependent variable). The linear relationship between these two variables can be represented by a straight line called regression line. The coefficients are estimated using the least-squares criterion, i.e., the best fit line has to be calculated that minimizes the sum of squared residuals (or "sum of squared errors").

The formula of the polynomial linear regression is almost similar to that of Simple Linear Regression. But in some datasets, it is hard to fit a straight line. Therefore, we use a polynomial function of a curve to fit all the data points.
Simple And Multiple Linear Regression: Simple linear regression has only one x and one y variable. Multiple linear regression has one y and two or more x variables. In simple linear regression there is a one-to-one relationship between the input variable and the output variable. But in multiple linear regression, as the name implies there is a many-to-one relationship, instead of just using one input variable, you use several.

Logistic Regression: It is a classification algorithm, that is used where the response variable is categorical. The idea of Logistic Regression is to find a relationship between features and probability of particular outcome. and for logistic regression, we calculate probability, i.e. y is the probability of a given variable x belonging to a certain class. Thus, it is obvious that the value of y should lie between 0 and 1. Here we Use Sigmoid Function. Logistic regression:  linear regression is leveraged when dependent variables are continuous, logistic regression is selected when the dependent variable is categorical, meaning they have binary outputs, such as "true" and "false" or "yes" and "no." While both regression models seek to understand relationships between data inputs, logistic regression is mainly used to solve binary classification problems, such as spam identification.

Support vector machines (SVM): A support vector machine is a popular supervised learning model developed by Vladimir Vapnik, used for both data classification and regression. It is typically leveraged for classification problems, constructing a hyperplane where the distance between two classes of data points is at its maximum. This hyperplane is known as the decision boundary, separating the classes of data points (e.g., oranges vs. apples) on either side of the plane. SVM is a discriminative classifier formally defined by a separating hyperplane. In other words, given labelled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two-dimensional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side. Before going to SVM work first know some general things:
1.	Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.
2.	Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane.
3.	Marginal Distance is the two parallel line that create w.r.t most near point of +ve and -ve to plane distance. we always maximise the marginal distance.
4.	The function of kernel is to take data as input and transform it into the required form. Different SVM algorithms use different types of kernel functions. These functions can be different types. For example, linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid.  Kernelling is SVMmapping data into a higher dimensional space, in such a way that can change a linearly inseparable dataset into a linearly separable dataset.

K-nearest neighbor: K-nearest neighbor, also known as the KNN algorithm, is a non-parametric algorithm that classifies data points based on their proximity and association to other available data. This algorithm assumes that similar data points can be found near each other. As a result, it seeks to calculate the distance between data points, usually through Euclidean distance, and then it assigns a category based on the most frequent category or average. Its ease of use and low calculation time make it a preferred algorithm by data scientists, but as the test dataset grows, the processing time lengthens, making it less appealing for classification tasks. KNN is typically used for recommendation engines and image recognition. KNN(K-Nearest Neighbors)Algorithm  is Supervised machine learning algorithm as target variable is known. Nonparametric as it does not make an assumption about the underlying data distribution pattern. It is Lazy algorithm as KNN does not have a training step. All data points will be used only at the time of prediction. It used for both Classification and Regression.
* What is K is K nearest neighbors?
- K is a number used to identify similar neighbors for the new data point.
- KNN takes K nearest neighbors to decide where new data point with belong to. This decision is based on feature 
  similarity.
* How do we choose the value of K?
 - We can evaluate accuracy of KNN classifier using K fold cross validation.
 - Using Elbow Method to Find Right Value OF K.
* Before Going to Working of KNN I am showing you to all distance that KNN Use. There are Three Types of Distance
 - Euclidean distance is square root of the sum of squared distance between two points. It is also known as L2 norm.
 - Manhattan distance is sum of absolute values of the differences between two points. Also known as L1 norm.
 - Hamming distance is used for categorical variables. In simple terms it tells us if two categorical variables are same 
   or not.
 - 
Naive bayes: Naive Bayes is classification approach that adopts the principle of class conditional independence from the Bayes Theorem. This means that the presence of one feature does not impact the presence of another in the probability of a given outcome, and each predictor has an equal effect on that result. There are three types of Naïve Bayes classifiers: Multinomial Naïve Bayes, Bernoulli Naïve Bayes, and Gaussian Naïve Bayes. This technique is primarily used in text classification, spam identification, and recommendation systems. Naive Bayes: It is a very popular Supervised Classification algorithm. This algorithm is called “Naive” because it makes a naive assumption that each feature is independent of other features which is not true in real life. As for the “Bayes” part, it refers to the statistician and philosopher, Thomas Bayes and the theorem named after him, Bayes’ theorem, which is the base for Naive Bayes Algorithm.
Random forest: Random Forest is another flexible supervised machine learning algorithm used for both classification and regression purposes. The "forest" references a collection of uncorrelated decision trees, which are then merged together to reduce variance and create more accurate data predictions.

