from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# FEATURE_PATH = '/yuanProject/XPath/AE_embedding/ae6_embed.npy'
# LABEL_PATH = '/yuanProject/XPath/AE_embedding/y.npy'
#
# features = np.load(FEATURE_PATH)
# labels = np.load(LABEL_PATH)
# colors = []
# for item in labels:
#     if item == 1:
#         colors.append('red')
#     else:
#         colors.append('blue')
# print(features.shape)
# tsne = TSNE(n_components=2, random_state=0)
# transformed_data = tsne.fit_transform(features)
# k = np.array(transformed_data)
# print(k.shape)
# plt.scatter(k[:, 0], k[:, 1], c=colors, zorder=10, s=4)
# plt.xlim([-25, 25])
# plt.ylim([-25, 25])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.xticks((-20, 0, 20))
# # plt.yticks((-20, 0, 20))
# plt.savefig('ae.png')



FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_query_features.npy'
LABEL_PATH = '/yuanProject/XPath/ds96/test/y_test_balanced.npy'

features = np.load(FEATURE_PATH)
labels = np.load(LABEL_PATH)
colors = []
for item in labels:
    if item == 1:
        colors.append('red')
    else:
        colors.append('blue')
print(features.shape)
tsne = TSNE(n_components=2, random_state=0)
transformed_data = tsne.fit_transform(features)
k = np.array(transformed_data)
print(k.shape)
plt.scatter(k[:, 0], k[:, 1], c=colors, zorder=10, s=4)
plt.xlim([-55, 55])
plt.ylim([-55, 55])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xticks((-20, 0, 20))
# plt.yticks((-20, 0, 20))
plt.savefig('relational.png')
