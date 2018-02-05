# Predicting Survival of Titanic Passengers Using Machine Learning
This repository includes my exploratory data analysis, feature engineering, model selection,
parameter selection, and final accuracy score for a popular Kaggle competition on
determining the survival rate of the Titanic passengers based on the their class,
sex, cabin fare, and other features.

## Data

I got the data from [Kaggle's Prediction Competition](https://www.kaggle.com/c/titanic),
in which other Kaggle users compete to create a machine learning model that will
most accurately predict which Titanic passengers survived based on data known for
all passengers.  The raw data given for each passenger included 'PassengerId',
'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
'Cabin', and 'Embarked'.

## Exploratory Data Analysis and Feature Engineering

Before doing any feature engineering, I wanted to see if there were any major predictors
for survival based on gender, age, class, fare, and number of family members.

![Survival Rate by Gender](images/survival_by_gender.png)
This graph shows that females survived in much higher proportions than males did,
which makes sense because women and children were given life boats first; therefore
over 70% of women survived while only 20% of men survived.

![Survival Rate by Class](images/survival_by_pclass.png)
This graph shows that 60% of passenger class 1 survived, over 40% of passenger class
2 survived, and a little over 20% of passenger class 3 survived.

![Survival/Death Rate by Age](images/histogram_survival_death_age.png)
This graph shows two histograms to compare visually those that survived versus those
who died across different age ranges; in some age ranges, more passengers survived
such as the children.

![Survival/Death Rate by Fare](images/histogram_survival_death_fare.png)
This graph shows that the cheaper your ticket, the greater likelihood your chance
of dying.

I found that there were null values in the 'Fare', 'Age', and 'Embarked' columns
where 'Embarked' is the location the passenger embarked from.  I dealt with the
null Fare values by  

  train["Age"].describe() #this shows that there are 714 ages but 814 rows, so there's missing ages

  # process_missing_vals fills in nulls with appropriate values and preps the Age column for binning
  for df in [train, holdout]:
      df = process_missing_vals(df)
  '''
  bin_column adds a column to show if passenger was an infant, child,
  teenager, young adult, adult, senior, or had a missing age
  '''
  age_cut_points = [-1,0,5,12,18,35,60,100]
  age_label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
  train = bin_column(train,'Age','Age_categories',age_cut_points,age_label_names)
  holdout = bin_column(holdout,'Age','Age_categories',age_cut_points,age_label_names)

  #bin_column adds a column to show if passenger paid between 0-12, 12-50, 50-100, or 100+
  fare_cut_points = [0,12,50,100,1000]
  fare_label_names = ["0-12","12-50","50-100","100+"]
  train = bin_column(train,'Fare','Fare_categories',fare_cut_points, fare_label_names)
  holdout = bin_column(holdout,'Fare','Fare_categories',fare_cut_points, fare_label_names)
  '''
  plot_pivot_survived(train, 'Age_categories')
  this graph shows that infants had a 70% chance of living, teenagers had the second highest
  chance at 40%, and seniors had the lowest chance at 20%
  '''
  #the extract_titles function takes each name and takes out Mr., Mrs., Master, etc.
  train = extract_titles(train)
  holdout = extract_titles(holdout)
  #the extract_cabin_type function takes the first letter of the cabin number
  train = extract_cabin_type(train)
  holdout = extract_cabin_type(holdout)
  #scaling numeric columns
  numeric_columns= ["SibSp","Parch","Fare"]
  for col in numeric_columns:
      train[col + "_scaled"] = minmax_scale(train[col])
      holdout[col + "_scaled"] = minmax_scale(holdout[col])
  '''
  plot_hist(train, 'SibSp') and plot_hist(train, 'Parch') show that you are much
  more likely to have died if you didn't have any siblings/spouses or parents/children;
  therefore we are going to combine 'SibSp' and 'Parch' so that we can then create
  a column to show whether or not a passenger was alone or not.
  '''
  for df in [train, holdout]:
      df['Sib_Parch'] = df['SibSp']+df['Parch']
      df['is_alone'] = df['Sib_Parch'].apply(is_alone)

  #creating dummies for categorical data
  categorical_columns = ['Pclass', 'Sex', 'Age_categories', 'Embarked', 'Fare_categories', 'Title', 'Cabin_type']
  for column in categorical_columns:
      train = create_dummies(train,column)
      holdout = create_dummies(holdout,column)

  #the select_features function uses a RandomForestClassifier to get the best features
  optimized_columns = select_features(train)
  grid_searched = select_model_tune(train, optimized_columns)
  #without taking out any columns before optimizing, we got a 83.7% on the RandomForestClassifier
  best_rf_model = grid_searched[2]['best_estimator']
  create_submission(best_rf_model,optimized_columns,filename = 'submission_final.csv')

  '''
  plot_correlation_heatmap(train)
  This heatmap showed that a lot of columns that I made from feature engineering
  are highly correlated with the features that they were made from, such as SibSp_scaled
  and SipSp, Parch and Parch_scaled, Fare and Fare_scaled, etc.  Also, I noticed
  that Title_Mr and Sex_male are extremely highly correlated,that Sex_male and
  Sex_female are extremely highly inversely correlated, and that the other titles
  show a high correlation with sex_male and sex_female.

  for df in [train,holdout]:
      df = df.drop(['Pclass','Name','Sex','Sex_male','Age','Age_categories',
                  'Fare_categories','Fare_categories','Cabin_type','SibSp',
                  'SibSp_scaled','Parch','Parch_scaled','Sib_Parch','Fare',
                  'Fare_scaled','Cabin','Embarked','Title'], axis = 1, inplace = True)

  optimized_columns = select_features(train)
  grid_searched = select_model_tune(train, optimized_columns)
  #this got a best of 82.9% on the RandomForestClassifier so I might choose to
  keep all the columns anyway, though I am afraid that I'm overfitting.
  '''



## Model Selection and Parameter Selection

## Results

## References
