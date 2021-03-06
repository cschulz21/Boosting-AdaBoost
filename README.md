# Boosting and AdaBoost

Normally when someone thinks of a machine learning algorithm they picture a Neural Network or some other model training on data to become the “expert” on the material and provide predictions. This approach takes one learner and performs a task. Just as we like to get second opinions in our day to day to get a better picture and more accurate results, ensemble methods were produced to get second, third, and fourth opinions to reduce error and bias. Ensemble methods take advantage of multiple learners in order to come to a decision. Because we are considering the results from multiple learners, they need not be as accurate as a single learner. In fact, we just need each of these “weak” learners to do better than random guessing. Boosting is a subset of the ensemble method that “boosts” the importance of data that we are misclassifying for subsequent learners to focus on. Boosting builds each learner one after another on what the learners before misclassified. 

### AdaBoost

To explain how boosting works, we will look at the most popular version of boosting, AdaBoost. AdaBoost was developed by Freund and Schapier in 1996 and it is a model that is often called the “best out-of-the-box classifier”. This is due to the very strong results and ease of implementation. To build the AdaBoost model we will take advantage of a large number of “weak” classifiers. One such “weak” classifier that is commonly used as the building block to AdaBoost is the decision stump. Decision stumps are the most basic form of a decision tree, in that they only have one test with 2 possible outcomes, true or false. 

<img src="decision_stump.png" alt="alt text" width="500">

In order to build an AdaBoost model on a set of labeled data, we first create a decision stump and try to predict what class each data point falls into. We look at how well the stump did at classifying, and, if the stump does well, we want our AdaBoost model to listen to this stump so we give it a high weight. If the stump does poorly, we give it a low weight, and, if the stump does worse than just randomly guessing, we give it a negative weight (essentially telling the model to do the opposite of what this stump says to do). 

We then look at which examples the stump did well on and which examples it misclassified. We want our next stump to “focus” on the examples we did not do well on. So, we create a weight for each data point, starting with all equal weights. If the stump classified a given data point correctly, we decrease its weight, and if the stump incorrectly classified the data point, we increase the weight. Something to note is this weight is related to the weight we gave the stump. For example, if the stump does really well and gets a high weight, the data it missed will get a higher weight because this data is “harder” to classify. 

Next, we create another stump based on the weighted data points. This stump will “focus” more on the data with higher weights and will favor classifying them over a data point with lower weight. Once we have our second decision stump, we evaluate it on all the data. This gives us our weight for this decision stump in our overall model. Just as before with the data, we adjust the weights of each data point based on if this second stump was able to classify it correctly or not, increasing the weights of the data we mislabeled and decreasing the weights of the data we labeled correctly. 

We then rinse and repeat, continuing this process over and over again. At each step we create a decision stump, train it on the weighted data, evaluate it, and adjust the weights of our data. We do this until we have a large number of these decision stumps (we can set the number of stumps we want our model to have). 

Now that we have a collection of stumps and a weight based on how well they were able to classify the data, we can generate our model for predictions. In order to evaluate a new data point and make a prediction, our AdaBoost model will consult each of our decision stumps and “listen” to their responses based on their respective weights. The better a stump did on our training data, the higher the weight it received and the more it will influence our predictions of new data. Essentially it is a weighted average of the outputs from the decision stumps. Because boosting models emphasise the data our classifier is not doing well on, outliers and mislabeled data will have a relatively large effect on our results, so spending the time to clean the data can be very beneficial.

### Demo

To see this model in action, I have included a file, abc.py, using the AdaBoost classifier from Scikit Learn [Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) on a toy-data set.

Once we have the data imported, we define what model we are using and how many estimators we would like on line 17 `abc = AdaBoostClassifier(n_estimators=50)`

We then train the model using the `fit` function on line 21 `model = abc.fit(X_train, y_train)`.

And we can make our predictions using the `predict` function as seen on line 25 `y_pred = model.predict(X_test)`

When run on the [Breast Cancer toy data set](https://scikit-learn.org/stable/datasets/index.html), without having to adjust any hyperparameters, we can achieve an accuracy of: `Accuracy: 0.9532163742690059`, showing just how powerful the boosting method can be and how easy it can be to implement.

### Summary

Boosting takes advantage of using a collection of "weak" learners in order to produce strong results. Each additional learner in these boosting models gets fed a subset of the data or a weighted data set to in order for the learner to prioritise the data that learners before it had trouble classifying. By combining these weak learners, we are able to increase accuracy and reduce bias, but these models are susceptible to outliers, so cleaning the data before training can greatly increase performance.

<sup>Reference: Schapire, Robert E. "The boosting approach to machine learning: An overview." Nonlinear estimation and classification. Springer, New York, NY, 2003.
</sup>
<sup>Special thanks to Professor Mike Izbicki https://izbicki.me/
</sup>
