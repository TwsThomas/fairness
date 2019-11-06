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
