import numpy as np

HE_QUERY_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96/he_query_features.npy'
PH_QUERY_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96/ph_query_features.npy'
HE_DATASET_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96/he_dataset_features.npy'
PH_DATASET_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96/ph_dataset_features.npy'


data = np.load(HE_QUERY_FEATURE_PATH)
print(len(data), data.shape)
data = np.load(HE_DATASET_FEATURE_PATH)
print(len(data), data.shape)