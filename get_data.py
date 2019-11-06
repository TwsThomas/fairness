import numpy as np
import copy
import pandas as pd


from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from joblib import Memory, Parallel, delayed
location = './cachedir'
memory = Memory(location, verbose=0)


@memory.cache
def get_adult(random_state=42):

    df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")
    # Or use the local copy:
    # df = pd.read_csv('../datasets/adult-census.csv')
    print(df.shape)
    target_name = "class"
    target = df[target_name].to_numpy()
    target = target == ' >50K'

    protected_groups = [df['sex']==' Female', df['sex'] ==' Male',
                        df['race'] == ' Black', df['race'] == ' White']

    data = df.drop(columns=[target_name, "fnlwgt"])

    gg = np.asarray(protected_groups).T
    print(gg.shape)
    data_train, data_test, target_train, target_test,\
         protected_train_T, protected_test_T = train_test_split(
        data, target, gg, random_state=random_state)

    # preprocessing steps
    binary_encoding_columns = ['sex']
    one_hot_encoding_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'native-country']
    scaling_columns = [
        'age', 'education-num', 'hours-per-week', 'capital-gain',
        'capital-loss']

    preprocessor = ColumnTransformer([
        ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
        ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
        one_hot_encoding_columns),
        ('standard-scaler', StandardScaler(), scaling_columns)])

    data_train = preprocessor.fit_transform(data_train)
    data_test = preprocessor.transform(data_test)

    protected_train, protected_test = protected_train_T.T, protected_test_T.T
    if len(protected_train) == 4:
        protected_train = protected_train[0]
        protected_test = protected_test[0]

    return data_train, data_test, target_train, target_test,\
           protected_train, protected_test




# def get_gk_adult(X, y=None, ohe=False):
#     # return the 4 protected group (sex & white/black)
    
#     if not(ohe):
#         Gk = [X['sex']=='Female', X['sex'] =='Male', X['race'] == 'Black', X['race'] == 'White']
#     else:
#         Gk = [X['sex_Female'], X['sex_Male'], X['race_Black'], X['race_White']]
    
#     Gk = [np.array(g, dtype=bool) for g in Gk]
#     return Gk


# def get_adult(return_df=False):
    
#     # read data

#     adult_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
#     df_train = pd.read_csv('data/adult.data', header=None, sep=', ', engine='python')
#     df_train.columns = adult_columns + ['y']
#     df_test = pd.read_csv('data/adult.test.txt', header=None, sep=', ', skiprows=1, engine='python')
#     df_test.columns = adult_columns + ['y']

#     X_train, y_train, X_test, y_test = df_train.drop('y', axis=1), df_train['y'], df_test.drop('y', axis=1), df_test['y']
#     y_train = y_train == '>50K'
#     y_test = y_test == '>50K.'
#     print('adult loaded')
#     print('X,y train/test shape', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#     if return_df:
#         return X_train, y_train, X_test, y_test
    
#     ## OHE
#     cat_columns = ['workclass', 'education', 'education-num', 'marital-status', 'occupation',
#                    'relationship', 'race', 'sex', 'native-country']
#     X_train_ohe = pd.get_dummies(X_train, columns=cat_columns)
#     # HACK for same OHE
#     X_test.at[X_test.index[-1], 'native-country'] = 'Holand-Netherlands'
#     X_test_ohe = pd.get_dummies(X_test, columns=cat_columns)
#     print('X_train_ohe X_test_ohe shape:')
#     print(X_train_ohe.shape, X_test_ohe.shape)

#     X_all = pd.get_dummies(pd.concat((X_train, X_test)), columns=cat_columns)
#     y_all = pd.concat((y_train, y_test))
#     shape_all = X_all.shape
#     # print(shape_all)
#     assert X_train_ohe.shape[1] == X_test_ohe.shape[1]
#     assert X_train_ohe.shape[1] == shape_all[1]
    
#     assert X_train_ohe.shape[0] == y_train.shape[0]
#     assert X_test_ohe.shape[0] == y_test.shape[0]
    
#     assert X_train_ohe.shape[0] + X_test_ohe.shape[0] == shape_all[0]
#     assert set(X_train_ohe.columns).difference(set(X_test_ohe.columns)) == set()
#     assert set(X_test_ohe.columns).difference(set(X_train_ohe.columns)) == set()


    # #sklearn version 
    # ct = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), cat_columns)])
    # X_train_ohe = ct.fit_transform(X_train)
    # print(X_train_ohe.shape)
    # X_test_ohe = ct.transform(X_test)
    
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_all, y_all, test_size=0.30, random_state=42)
#     X_train_ohe, X_test_ohe, y_train, y_test = train_test_split(
#         X_all, y_all, test_size=0.30, random_state=42)

    return X_train_ohe, y_train, X_test_ohe, y_test


# def get_bank(return_df=False, path='data/bank/bank-full.csv'):
#     print('load ', path)
#     df = pd.read_csv(path, sep=";")
#     y = df['y'] == 'yes'
#     X = pd.get_dummies(df.drop('y', axis=1))

#     if return_df:
#         return X, y
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.30, random_state=42)
#     # print('age quantile =', np.quantile(X['age'], [i/6 for i in range(1,6)]))
#     return X_train, y_train, X_test, y_test


# def get_gk_bank(X, y=None):
#     # return the 5 protected group (by age)
    
#     age_q = list(np.quantile(X['age'], [i/6 for i in range(1,6)]))
#     print('age quantile =', age_q)
    
#     age_bins = [0] + age_q + [max(X['age'])+1]
#     Gk = []
#     for min_, max_ in zip(age_bins[:-1], age_bins[1:]):
#         # print(min_, max_)
#         Gk.append(np.array((X['age'] >= min_) & (X['age'] < max_)).copy())
        
#     assert X.shape[0] == sum([x.sum() for x in Gk])
#     Gk = [np.array(g, dtype=bool) for g in Gk]
#     return Gk