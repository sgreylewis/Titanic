import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

def plot_pivot_survived(df, index):
    pivot = df.pivot_table(index=index,values="Survived")
    plt.axhspan(.3, .6, alpha=0.2, color='red')
    pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))
    pivot.plot.bar()
    plt.show()

def plot_hist(df, column):
    survived = train[train["Survived"] == 1]
    died = train[train["Survived"] == 0]
    survived[column].plot.hist(alpha=0.5,color='red',bins=10)
    died[column].plot.hist(alpha=0.5,color='blue',bins=10)
    plt.legend(['Survived','Died'])
    plt.show()

def plot_correlation_heatmap(df):
    corr = df.corr() #computes pairwise correlation of columns
    sns.set(style="white") #sets aesthetic parameters in one step
    mask = np.zeros_like(corr, dtype=np.bool) #creates an array with the same size as corr, fills it with zeros, and makes the dtypes boolean
    mask[np.triu_indices_from(mask)] = True #fills the indices for the upper-triangle of mask with true
    f, ax = plt.subplots(figsize=(11, 7))
    cmap = sns.diverging_palette(220, 10, as_cmap=True) #returns a matplotlib color palette object
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
    square=True, yticklabels = True, linewidths=.5, cbar = True, cbar_kws={"shrink": .5})
    #since mask is true, data will not be shown in cells where mask is True
    #the cmap is the mapping from data values to color space
    #vmax are the values to anchor the colormap
    #center is the value at which to center the colormap when plotting divergent data
    #square sets the axes aspect to equal so each cell will be square-shaped
    #linewidths is the width of the lines that will divide each cell
    #cbar_kws is the keyword arguments for fig.colorbar so it shrinks the colorbar on the right to half its size
    plt.yticks(fontsize = 10, rotation='horizontal') #makes the ylabels horizontal
    plt.xticks(fontsize = 10, rotation=90) #makes the xlabels vertical
    plt.show()

def process_missing_vals(df):
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Age"] = df["Age"].fillna(-0.5)
    return df

def bin_column(df, column, new_column, cut_points, label_names):
    df[new_column] = pd.cut(df[column],cut_points,labels=label_names)
    return df

def extract_titles(df):
    titles = {
    "Mr":"Mr","Mme":"Mrs","Ms":"Mrs","Mrs":"Mrs","Master":"Master","Mlle":"Miss",
    "Miss":"Miss","Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer",
    "Rev":"Officer","Jonkheer":"Royalty","Don":"Royalty","Sir":"Royalty",
    "Countess":"Royalty","Dona":"Royalty","Lady":"Royalty"}
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def extract_cabin_type(df):
    df['Cabin_type'] = df['Cabin'].str[0]
    df['Cabin_type'] = df['Cabin_type'].fillna('Unknown')
    return df

def is_alone(row):
    if row == 0:
        return 0
    else:
        return 1

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

def select_features(df):
    #remove non-numeric columns and columns that have null values
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    df = df.dropna(axis=1, how='any')
    all_X = df.drop(['PassengerId', 'Survived'],axis=1)
    all_y = df["Survived"]
    clf = RandomForestClassifier(random_state=1)
    selector = RFECV(clf, cv = 10)
    selector.fit(all_X,all_y)
    optimized_columns = list(all_X.columns[selector.support_])
    return optimized_columns

def select_model_tune(df, columns):
    all_X = df[columns]
    all_y = df['Survived']
    list_of_dicts = [{"name": "LogisticRegression","estimator": LogisticRegression(),
    "hyperparameters":{"solver": ['newton-cg', 'lbfgs', 'liblinear']}},
     {"name": "KNeighborsClassifier","estimator": KNeighborsClassifier(),
    "hyperparameters":{"n_neighbors": range(1,20,2),"weights": ["distance", "uniform"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],"p": [1,2]}},
    {"name": "RandomForestClassifier","estimator": RandomForestClassifier(),
    "hyperparameters":{"n_estimators": [4,6,9],"criterion": ['entropy','gini'],
        'max_depth':[2,5,10],"max_features": ['log2', 'sqrt'],
        'min_samples_leaf': [1,5,8],'min_samples_split': [2,3,5]}}]
    for dictionary in list_of_dicts:
        print ('Model: {0}'.format(dictionary['name']))
        grid = GridSearchCV(dictionary['estimator'], param_grid=dictionary['hyperparameters'], cv=10)
        grid.fit(all_X, all_y)
        dictionary['best_params'] = grid.best_params_
        dictionary['best_score'] = grid.best_score_
        dictionary['best_estimator'] = grid.best_estimator_
        print('Parameters: {0}'.format(dictionary['best_params']))
        print('Score: {0}'.format(dictionary['best_score']))
    return list_of_dicts

def create_submission(model, columns, filename='submission.csv'):
    holdout_data = holdout[columns]
    predictions = model.predict(holdout_data)
    holdout_ids = holdout["PassengerId"]
    submission_df = {"PassengerId": holdout_ids,"Survived": predictions}
    submission = pd.DataFrame(submission_df)
    submission.to_csv(filename, index = False)

if __name__ == '__main__':
    holdout = pd.read_csv("test.csv")
    train = pd.read_csv('train.csv')
    '''
    #plot_pivot_survived(train, 'Sex')
    this shows that females survived in much higher proportions than males did, which
    makes sense because women and children were given life boats first
    over 70% of women survived while only 20% of men survived

    plot_pivot_survived(train, 'Pclass')
    this shows that 60% of Pclass 1 survived, over 40% of Pclass 2 survived, and
    a little over 20% of Pclass 3 survived
    train["Age"].describe() #this shows that there are 714 ages but 814 rows, so there's missing ages

    plot_hist(train, "Age")
    this shows two histograms to compare visually those that survived versus those
    who died across different age ranges; in some age ranges, more passengers survived
    such as the children

    plot_hist(train, 'Fare')
    This shows that the cheaper your ticket, the greater likelihood your chance
    of dying.
    '''

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

    #Remove the columns Pclass_2, Age_categories_Teenager, Fare_categories_12-50,
    #TItle_Master, Cabin_type_A.

    best_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Age_categories_Missing',
     'Age_categories_Infant', 'Age_categories_Young Adult', 'Fare_categories_12-50',
     'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Cabin_type_D', 'Cabin_type_E',
     'Cabin_type_Unknown', 'Sex_female', 'Sex_male', 'is_alone']
    #grid_searched = select_model_tune(train, best_columns)
    #this gave a best of 83.8%

    relevant_columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_categories_0-12',
       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',
       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',
       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',
       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown']


    new_columns = ['Age_categories_Missing', 'Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Young Adult',
           'Age_categories_Adult', 'Age_categories_Senior', 'Pclass_1', 'Pclass_3',
           'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',
           'Parch_scaled', 'Fare_categories_0-12', 'Fare_categories_50-100',
           'Fare_categories_100+', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
           'Title_Officer', 'Title_Royalty', 'Cabin_type_B', 'Cabin_type_C',
           'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',
           'Cabin_type_T', 'Cabin_type_Unknown']
