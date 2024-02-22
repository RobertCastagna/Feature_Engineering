# lib
import talib as ta
import numpy as np
import pandas as pd

#ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

#feature
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF

# custom
import tautil

import warnings
warnings.filterwarnings(action='ignore')

import yfinance as yf

# Accuracy function for price prediction
def get_acc(close, feature, model, days=5, kfold=3):
    
    mom = np.sign(close.shift(-days)-close)
    y = mom.copy() # choose y
    raw_X = feature.copy() # choose X

    if model == 'xgb':
        y = y.map({-1: 0, 1: 1})

    tmp_df = raw_X.join(y).dropna()
    raw_X=tmp_df.iloc[:,:-1]
    y=tmp_df.iloc[:,-1]
    t1 = pd.Series(y.index, index=y.index) # if y = mom

    # CV
    cv = KFold(kfold)

    # Scaling
    scaler = preprocessing.MinMaxScaler((0,1))
    scaler.fit(raw_X)
    scaled_X = pd.DataFrame(scaler.transform(raw_X), index=raw_X.index, columns=raw_X.columns)
    X = scaled_X

    # Choose model
    rfc = RandomForestClassifier(n_estimators = 200, criterion='entropy', class_weight='balanced_subsample', bootstrap=True)
    svc = SVC(probability=True)
    xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.5, eval_metric='logloss')


    if model == 'rf':
        clf = rfc
    elif model == 'svm':
        clf = svc
    elif model == 'xgb':
        clf = xgb

    accs=[]
    for train, test in cv.split(X, y):
        clf.fit(X.iloc[train],y.iloc[train])
        y_true = y.iloc[test]
        y_pred = clf.predict(X.iloc[test])
        accs.append(accuracy_score(y_true,y_pred))
        
    feature_importances = clf.feature_importances_
    importances_dict = dict(zip(raw_X.columns, feature_importances))
    for name, importance in sorted(importances_dict.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {importance}")

    return np.mean(accs)


df_ = yf.download('GOOG','2020-1-1','2024-1-1')
df = tautil.ohlcv(df_)
windows = [5,7,15,20]


TA_all = tautil.get_stationary_ta_windows(df, windows).dropna()

# st_test = tautil.ta_stationary_test(TA)
# print(st_test)

TA = tautil.remove_non_stationary_ta(TA_all)
print(TA.describe())
print('Non-stationary feature(s) removed: ', set(TA_all).symmetric_difference(set(TA)))


# Original features
feature = TA.copy()
close = df.close
raw_acc = get_acc(close, feature, model='xgb')
print("Original features acc score: ", raw_acc)


# -------------------------- Component Decomposition (PCA, NMF)    

# scaler = preprocessing.StandardScaler()
# TA_ = scaler.fit_transform(TA)

# mask = np.isnan(TA_)
# TA_ = TA_[~mask.any(axis=1)]

# # Linear PCA
# pca = PCA(n_components=3)
# pca_TA = pca.fit_transform(TA_)
# feature_weights = pca.components_
# pca_TA = pd.DataFrame(pca_TA, index=TA.dropna().index)

# feature = pca_TA.copy()
# close = df.close
# pca_acc = get_acc(close, feature, model='rf')
# print("PCA features acc score: ", pca_acc)

# # Non-negative matrix factorization
# nmf = NMF(n_components=3)
# TA_ = np.where(TA_ < 0, 0, TA_)
# nmf_TA = nmf.fit_transform(TA_)
# feature_weights = nmf.components_
# nmf_TA = pd.DataFrame(nmf_TA, index=TA.dropna().index)

# feature = nmf_TA.copy()
# close = df.close
# nmf_acc = get_acc(close, feature, label='dir', model='rf')
# print("NMF features acc score: ", nmf_acc)
