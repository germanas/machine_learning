
#Identify Fraud from Enron Email
##I. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? [relevant rubric items: “data exploration”, “outlier investigation”]
The goal of this project was to figure out the persons of interest using machine learning. Machine learning was usefull here because we can put many features such as salary, emails to poi, bonuses and predict if the person is POI or not based on these features. I got hold of excerpt of the data from enron corpus, and tried to create a model to predict persons of interest. Exploring the data set I got these statistics:
Number of total datapoints: 146
Number of features for each datapoint: 21
Number of persons of interest in this dataset: 18
Number of other people in this dataset: 128
Total feature values missing: 1352
Total feature values: 3190
The percentage compared to all values: 42.3824451411
The dataset is very limited and contains about 42% of missing values. I couldn't remove much of outliers in statistical way by calculating the quartile. I plotted the features to find most extreme outliers and I found 1 that needed removal. The key "TOTAL" had the totals of all salaries and it skewed the data. I removed it by poping "Total" from the dict.
##II. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]
I ended up using "SelectKBest" for feature selection. I used parameter k='all' so that the algorithm would check all of the features. After my parameter tuning I found out that the k-score with StratifiedShuffleSplit is giving the best results. So I used 10 features in the final clasifier.
salary score is: 18.575703268
bonus score is: 21.0600017075
long_term_incentive score is: 10.0724545294
deferred_income score is: 11.5955476597
deferral_payments score is: 0.21705893034
loan_advances score is: 7.24273039654
other score is: 4.2049708583
expenses score is: 6.23420114051
director_fees score is: 2.10765594328
total_payments score is: 8.86672153711
exercised_stock_options score is: 25.0975415287
restricted_stock score is: 9.34670079105
restricted_stock_deferred score is: 0.0649843117237
total_stock_value score is: 24.4676540475
to_messages score is: 1.69882434858
from_messages score is: 0.164164498234
from_this_person_to_poi score is: 2.42650812724
from_poi_to_this_person score is: 5.34494152315
ratio_of_poi_emails score is: 0.423244226689
I created a new feature called "ratio of poi_to and poi_from". I wanted to see what is the ratio of sent and received emails from poi and maybe find some patterns in this data. I found that my newly created feature had such a low importance of 0.42 that when I use GridSearchCV to find the best parameters, it selects K - 5 or K - 10 and outomaticaly dismisses the new feature. So there is no point in using this feature to identify POI.
I have noticed that the scale of some of my features are really different. For example the scale of emails sent is from 0 to about 500, but the scale for salary can go up to 1 000 000. To compare these features, I had to use feature scaling, to scale down the salary feature. I ended up using MinMaxScaler() in my code.
##III. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [relevant rubric item: “pick an algorithm”]
My features were labeled so I used these algorithms to test the accuracies: Naive bayes, decision trees and K nearest neigbor. I ended up using naive bayes (gaussianNB) algorithm because it had the best performance when I tested using tester.py. From my testing I got these results:
GaussianNb: Accuracy: 0.76800 Precision: 0.19544 Recall: 0.16300 F1: 0.17775 F2: 0.16860
DecisionTree: Accuracy: 0.75677 Precision: 0.24428 Recall: 0.27750 F1: 0.25983 F2: 0.27015
KNeigbors: Accuracy: 0.81231 Precision: 0.00673 Recall: 0.00150 F1: 0.00245 F2: 0.00178
I was actualy surprised that Kneigbors got really low scores, I was expecting it to be higher.
##IV. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). [relevant rubric item: “tune the algorithm”]
Parameter tuning is necessary to improve the performance of my machine learning algorithm by improving accuracy and preccision. To find out which parameters was the best I automated the tuning by using GridSearchCV function. This takes different parameters and returns those parameters who perform the best. If I wouldn't have had tuned the parameters and used the defaults, i would have gone way worse results in accuracy and precission. For the best scores I used GaussianNB classifier with feature scaling and and 10 best perforimg features using SelectKBest.
I was playing with different parameters in my gridsearch but I was getting precission: 0.23538. After reading the forums I found out that I should use StratifiedShuffleSplit to split the train and test data more equaly. That worked and the best parameters of my gridsearchcv giving the best scores was: using MinMaxScaler, SelectKBest(k=10), classifier - GausianNB and StratifiedShuffleSplit.
Performance before tuning -
Accuracy: 0.74780 Precision: 0.23538 Recall: 0.39650 F1: 0.29540 F2: 0.34876
After tuning -
Accuracy: 0.84427 Precision: 0.39855 Recall: 0.33000 F1: 0.36105 F2: 0.34176
##V. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? [relevant rubric item: “validation strategy”]
Validation is a strategy to separate your data to train and test sets so the results would not be overfitted and the most accurate and not corrupted by the training. The classic mistake when using a small sample size when every data point is important for model building, so spliting the data could separate the data unevenly, and that's why I will use stratifiedShuffleSplit.
I validated my data using StratifiedShuffleSplit and train_test_split. Using StratifiedShuffleSplit was helpfull because of my small dataset with a small number of POI's because of that the data is skewed towards non-POI. This fixes these issues and picks data for testing more carefully. The full definition of StratifiedShuffleSplit is "This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class."
##VI. Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
My selected evaluation metrics are precission and recall.
Precission: This is the ratio of predicted labels (POI) that were actualy the persons of interest.
$$Precision= \dfrac{TruePositive}{TruePositive+FalsePositive} $$
Recall: This ratio tells how many POI's were identified that are actual POI's and how many POI's there actualy are in the dataset.
$$Recall= \dfrac{TruePositive}{TruePositive+FalseNegative} $$
In my case using tester.py, I got precission of 0.39855 and recall of 0.33000
