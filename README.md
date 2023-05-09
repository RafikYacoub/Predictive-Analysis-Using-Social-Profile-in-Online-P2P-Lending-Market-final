# Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Market by TEAM B

Members of Team B:
Rafik Yacoub (team lead)
Yasmine Elegily (team co-lead)
Nada Metwaly
Mohamed Selim
Faiza Sultana
Priyal Jain
Manjula
Gokulakrishnan J
Gufran
Gourav Saha

Mostly the data are related to the amount of loan, dates, rates and other charactrestics about the borrowers.
In this project , we have done the predictive analysis of these parameter such as (** Loan Status ** , ** Borrower Rate**) along with other desired parameter 
in an Online P2P lending market related to determinants of performance predictability by conceptualizing its financial benefits to make futher predictions whether one can successfully predict which loans will default. 
There is thus a significant financial incentive to accurately predicting which of the loans would eventually default or not.Prosper loans pay pretty hefty interest 
rates to their creditors. 

## Understanding the Datasets:

The Main Dataset contains 113,937 loans with 81 variables on each loan, including loan amount, borrower rate (or interest rate), current loan status, 
borrower income, and many others.Out of these 113937 rows and 81 columns, We are going to select only the columns that are related to our data exploration and 
chose 15 columns to investigate further such as (**amount of loan, dates, rates and other charactrestics about the borrowers**).
Some columns are numeric but we also have categorical variables contains both (**ordinal and nominal**) and datetime variables.

#Main features of Interest in Dataset:

Now we  will work with features such as (**Borrower Rate, Borrower APR, Prosper Score, Credit Score, Original Loan Amount, Monthly Payment, Borrower Occupation, Borrower State and others if needed**).

> There are some important features to look at including:

*   BorrowerAPR: The Borrower's Annual Percentage Rate (APR) for the loan.
*   ProsperScore: A custom risk score built using historical Prosper data. The score ranges from 1-10, with 10 being the best or lowest risk score. 
    Applicable for loans originated after July 2009.
*   LoanOriginationDate: The date the loan was originated.
*   LenderYield: The Lender yield on the loan. Lender yield is equal to the interest rate on the loan less the servicing fee.

Other Features that will help us to support investigation in the Prosper Loan Data features are:

**Loan Status** and **Employment Status** will have a strong impact on loan and the features we are trying to explore further are 
 also the Monthly Income will play a role here and the Term (lenght of the loan) may have an effect.
" Borrower APR is normally distrubuted with the peak between 15 and 20 percent in addition we have some increase in the 35 percent"
"Prosper Scores are almost normally distributed and values 4, 6 and 8 are the most common."
"Lender Yield is nomarlly distributed with most of the values between 0.1 and 0.2 and we notice an increase at 0.3 "

## Preprocessing and Sentiment Analysis.

Beginning with categorical data, for now let's just fill the NaN values with "Unknown".
Next, about 20 loans are missing an APR value. Because APR is equal to the borrower rate + fees, let's calculate the median difference between the two, and 
add that value to the borrower rate of our data points missing an APR.
The numeric ProsperRating and the ProsperScore NaNs can both be replaced with median values.
It seems that most of our remaining Null values fall into four groups:
1) DebtToIncomeRatio, which strikes me as a potentially very useful feature. Let's take a look at what's going on there and do our best to reconstruct or substitute the missing values.
2) ScorexChangeAtTimeOfListing, which is the difference between the borrower's credit score when it was reviewed for this loan, versus the score last time they took a Prosper loan. We'll have to think about how we can deal with that, because it's an interesting potential feature.
2) Data dealing with the debtor's Prosper history, which we can fill with 0s to represent a lack of such history.
3) LoanFirstDefaultedCycle, which we are actually going to drop entirely very shortly for reasons to be explained.

First though, it's a bit annoying having to constantly refer to our variable definitions to understand the listing category, so before we interpret this, let's change our numeric values to the actual category names. This will also be useful, because the numeric values imply some sort of false ordinality, and we should really handle this like a categorical variable.

Let's scale our values to adapt them to some of our classifiers. I'm going to scale the values to a range between 0 and 1, as this will preserve our important 0 values, as well as having the added benefit of "robustness to very small standard deviations of features"
X_train_reduce50 = SelectPercentile(percentile=50).fit_transform(X_train_scaled, y_train)
X_test_reduce50 = SelectPercentile(percentile=50).fit_transform(X_test_scaled, y_test)

X_train_reduce10 = SelectPercentile().fit_transform(X_train_scaled, y_train)
X_test_reduce10 = SelectPercentile().fit_transform(X_test_scaled, y_test)

**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, but we want our test data to be a completely new and a surprise set for our model.

## Preprocessing again.

After EDA through Univariate, Bi-variate and Multivariate analysis we needed a to split the dataset for modelling process. And this section deals with splitting the dataset and the whole process of Feature Engineering. 

We start by dropping few variables starting with PercentFunded, EmployeeEffectiveYield etc. Both BorrowerAPR and LenderYield are fee-containing variants of BorrowerRate, making them superfluous. Estimates are useless since occupation is a broad and challenging column to employ; it is up to us to outperform them.
We further create a variable for ease of model called "first_credit_year" which basically contains FirstRecordedCreditLine. This allows us to proceed further. We convert categorical variable into dummy variable for "WageGroup". To obtain MI for both discrete and continuous features individually, we have to obtain all discrete and continuous features.

# Label Encodinng 

After dropping few variables and creating new ones, we then get to Label encoding part where we import the function and encode the column "CreditGrade" first and transform it. All of the category data are converted into numerical data using the label encoding technique. The loan status field is then likewise converted to binary data. ExtraTreesClassifier is a method for determining the importance of each independent feature in relation to a dependent feature. Feature importance will assign a score to each feature of your data, with the higher the score indicating that the feature is more important or relevant to the output variable.

# PCA 

After this step, mutual info classifier was performed for feature selection and selected the most important features. Besides using PCA as a data preparation technique, we can also use it to help visualize data. With the data visualized, it is easier for us to get some insights and decide on the next step in our machine-learning models. The aim of this step is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis. Here I standardize all the main features for the PCA and create Principal components thereby converting it to a data frame. We checked the model after performing PCA and it had an accuracy of 99.% with respect to the features.






##Exploratory data Analysis (EDA)


##Data Cleaning
The data cleaning or data scrubbing, is the process of fixing incorrect, incomplete, duplicate or otherwise erroneous data in a data set. It involves identifying data errors and then changing, updating or removing data to correct them. Data cleansing improves data quality and helps provide more accurate, consistent and reliable information for decision-making in an organization.

A few of the dataset's columns are listed below:
ListingNumber
ListingCreationDate
CreditGrade
Term
LoanStatus
ClosedDate
BorrowerAPR
BorrowerRate
LenderYield
EstimatedEffectiveYield
EstimatedLoss
EstimatedReturn
ProsperRating
This data table stored as 'df' has 113937 rows and 81 columns. Following that, because we have 81 variables and some of the cells may have missing data, I will remove the columns with more than 80% NA's. Cleaning the dataset
●	Our dataset contains null values mainly in the form of "?" however because pandas cannot read these values, we will convert them to np.nan form.


Univariate Plots Section
A univariate plot depicts and summarizes the data's distribution. Individual observations are displayed on a dot plot, also known as a strip plot. A box plot depicts the data in five numbers: minimum, first quartile, median, third quartile, and maximum.
Research Question 1 : What are the most number of borrowers Credit Grade?
●	Check the univariate relationship of Credit Grade

sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.countplot(y='CreditGrade',data=df)
 

df['LoanStatus'].hist(bins=100)
Since there are so much low Credit Grade such as C and D , does it lead to a higher amount of delinquency?
●	Check the univariate relationship of Loan Status 
What is the highest number of BorrowerRate?
●	Check the univariate relationship of Borrower rate
df['BorrowerRate'].hist(bins=100)
 
d Since the highest number of Borrower Rate is between 0.1 and 0.2, does the highest number of Lender Yield is between 0.1 and 0.2?
●	Check the univariate relationship of Lender Yield on Loan
df['LenderYield'].hist(bins=100)
 
Bivariate Plots Section
Bivariate analysis is a type of statistical analysis in which two variables are compared to one another. One variable will be dependent, while the other will be independent. X and Y represent the variables. The differences between the two variables are analyzed to determine the extent to which the change has occurred.
Discuss some of the relationships you discovered during this phase of the investigation. What differences did the feature(s) of interest have with other features in the dataset?
My main point of interest was the borrower rate, which had a strong correlation with the Prosper Rating. Borrower rate increased linearly as rating decreased.Listed below are some good example of research questions that have been subjected to bivariate analysis.
Is the Credit Grade really accurate? Does higher Credit Grade leads to higher Monthly Loan Payment? As for Higher Credit Grade we mean from Grade AA to B.
●	Check the Bivariate Relationship between Credit Card and MonthlyLoan Payment.

base_color = sns.color_palette()[3]
plt.figure(figsize = [20, 5])
plt.subplot(1, 2, 2)
sns.boxplot(data=df,x='CreditGrade',y='MonthlyLoanPayment',color=base_color);
plt.xlabel('CreditGrade');
plt.ylabel('Monthly Loan Payment');
plt.title(' Relationship between Credit Grade and MonthlyLoan Payment');
 
●	Check the Bivariate Relationship between BorrowerState and LoanStatus
 
Multivariate Plots
Multivariate analysis is traditionally defined as the statistical study of experiments in which multiple measurements are taken on each experimental unit and the relationship between multivariate measurements and their structure is critical for understanding the experiment. So, let's look at an example of a research question that we solved using multivariate plots and matplotlib.
Now we know the Credit Grade is accurate and is a tool that is used by the organization in determining the person’s creditworthiness. Now we need to understand does the ProsperScore, the custom built risk assessment system is being used in determining borrower’s rate?
●	Check the Multivariate Relationship between BorrowerRate and BorrowerAPR.

plt.figure(figsize = [20, 5])
plt.subplot(1, 2, 1)
plt.scatter(data=df,x='BorrowerRate',y='BorrowerAPR',color=base_color);
plt.xlabel('Borrower Rate');
plt.ylabel('Borrower APR');
plt.title('Relationship Between Borrower Rate and BorrowerAPR');
 

g = sb.FacetGrid(data = df, col = 'Term', height = 5,
                margin_titles = True)
g.map(plt.scatter, 'BorrowerRate', 'BorrowerAPR');
plt.colorbar()
 
From a theoretical standpoint, if the higher ProsperScore leads to lower Borrower Rate and Borrower Annual Percentage Rate that means the Prosper Score is being used alongside the Credit Grade in determining a person’s creditworthiness.



## Model Building
When building a machine learning model for P2P lending, it is important to carefully select and engineer features that are relevant to the problem at hand. This can involve extracting information from text data, such as loan descriptions or borrower comments, as well as combining information from multiple sources, such as credit scores and employment history.

Once the features have been selected, it is important to choose an appropriate algorithm or model to use for prediction. Some common models used in P2P lending include decision trees, random forests, and gradient boosting algorithms. These models can be used to make binary predictions, such as whether a loan will default or not, or to estimate the probability of default.

During the training process, it is important to carefully evaluate the performance of the model and adjust the hyperparameters to optimize performance. This may involve using cross-validation techniques to test the model on different subsets of the data, or tuning the model's hyperparameters using grid search or other optimization techniques.

It is also important to carefully consider the balance between model accuracy and interpretability. While more complex models may achieve higher accuracy, they may be more difficult to interpret and explain to stakeholders. This is particularly important in the P2P lending context, where transparency and fairness are important considerations.

Using the Model

The models used are the following:
qda = QuadraticDiscriminantAnalysis()

rf = RandomForestClassifier(random_state=0)

LR = LogisticRegression(penalty='l2')

reg = DecisionTreeRegressor()

Once a machine learning model has been trained and tested, it can be used to make predictions on new data. In the context of P2P lending, this might involve predicting the probability of default for a new loan application based on the borrower's characteristics and loan details.

To use the model, users typically need to provide input data in a format that is compatible with the model's requirements. This may involve preprocessing the data to ensure it is in the correct format or extracting relevant features from text data.

After the model has made a prediction, it is important to carefully interpret and evaluate the output. This may involve considering the probability of default in the context of the overall risk portfolio, or comparing the predicted probability to established risk thresholds.

Finally, it is important to continuously monitor and evaluate the performance of the model over time. This may involve tracking model accuracy, evaluating the model on new data, or updating the model with new features or data sources as they become available.

Overall, building and using machine learning models for P2P lending requires careful consideration of the relevant features, algorithms, and performance metrics, as well as ongoing monitoring and evaluation to ensure the model remains accurate and fair over time.

## Deployment 
The deployment part consists of three:
•	Creating the User Interface with html and CSS.
•	Connecting the user interface with Flask.
•	Deploying the application in Amazon EC2.

## Building the app
In order to build our app we needed to create 2 essential files along the ‘regression_model.joblib’ and the ‘classification_model.joblib’ files which contain our models.
•	index.html file: Using simple HTML, and embedded CSS; we created a user interface.
The html file simply has the features we need to ask the user for and inputs them. It also connects the user’s input to our flaskapp.py file to make a prediction.

•	Flaskapp.py file: We wrote a simple flask python code for our app.
Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications in Python easier.

## Predictions
After the user hits submit, the output will be printed at the end of the page.
The first output is the prediction of the classification model, which states whether this user is accepted or rejected for a loan application.
Then, the second output are the regression model predictions.
•	Preferred Return on investment (PROI)
•	Eligible loan amount (ELA)
•	Equated Monthly Instalments (EMI)
Finally, the calculation of the financial risk percentage is printed.

## Deploying on AWS.
We then deployed our flask app to an Amazon EC2 instance. You can access the app by following this link: http://16.16.97.69:12025/ .




