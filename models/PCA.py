import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

dataset = pd.read_csv('../datasets/backdoor_webshells_features_full.csv')
# dataset = dataset.drop(['Filename'], axis=1)
dataset = dataset.drop(['Super_Nasty_Sig_Count'], axis=1)
dataset = dataset.drop(['CloudflareBypass'], axis=1)
dataset = dataset.drop(['SuspiciousEncoding'], axis=1)

dataset_norm = dataset.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

dataset_np = dataset_norm.values
print(dataset_np.shape)
# print(dataset_np[0])


pca = PCA(0.95)
transformed_feat = pca.fit_transform(dataset_np)
# pritn(dataset)

transformed_dataset = np.hstack((transformed_feat, dataset_np[:,-1].reshape(dataset_np.shape[0],1).astype(int)))

print(transformed_dataset[0:3])
print(pca.explained_variance_ratio_) 

np.savetxt('../datasets/webshells_features_full_pca.csv', transformed_dataset, delimiter=',')

