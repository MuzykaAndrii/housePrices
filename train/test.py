import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


house_prices_file_path = './train.csv'
X = pd.read_csv(house_prices_file_path)

# drop rows with missing target
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

cols_to_use = ['MSSubClass', 'MSZoning', 'LotArea', 'Alley', 'LotShape', 'Condition1', 'BldgType', 'OverallCond',
                'YearBuilt', 'RoofStyle', 'RoofMatl', 'ExterCond', 'Heating', 'Electrical', 'SalePrice']

X = X[cols_to_use].copy()

# in production mode we should use all columns, to simplify web prediction interface we will use several columns
# y means prediction target column
y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)

# select list of column names with low cardinality
categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() < 10 and
                    X[cname].dtype == "object"]

# select numerical cols
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X = X[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    return scores.mean()

# results = dict()
# count_of_n_estimators = range(50, 1001, 50)

# for num_of_estimatros in count_of_n_estimators:
#     results[num_of_estimatros] = get_score(num_of_estimatros)

# # 450 - best count of estimators, 5 - best count of cross validation frames

# import matplotlib.pyplot as plt

# plt.plot(list(results.keys()), list(results.values()))
# plt.show()

# final_model = RandomForestRegressor(n_estimators=450, random_state=0)

# final_pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('model', final_model)
#     ])

# final_pipeline.fit(X, y)

# with open("titanic_pipeline.pickle", "wb") as outfile:
#     pickle.dump(final_pipeline, outfile)