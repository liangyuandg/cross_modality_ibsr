import os
import numpy as np


PATH = '/yuanProject/XPath/heatmap'

for sub_dir in os.listdir(PATH):
    for item in os.listdir(os.path.join(PATH, sub_dir)):
        if 'annotated' in item and '.npy' in item:
            gt = np.load(os.path.join(PATH, sub_dir, item))
        elif 'pred_prob' in item and '.npy' in item:
            pred = np.load(os.path.join(PATH, sub_dir, item))
    print(sub_dir, np.sum(gt==1), pred.shape[0]*pred.shape[1])

    # retrival hit/miss
    flattened_pred = pred.flatten()
    for i in range(1, 11):
        index = np.where(pred == flattened_pred[np.argsort(flattened_pred)[-i]])
        if gt[index[0], index[1]] == 1:
            print('hit')
        else:
            print('miss')

    # real rank
    flattened_pred_sorted = flattened_pred.copy()
    flattened_pred_sorted = -np.sort(-flattened_pred_sorted)
    indexes = np.where(gt == 1)
    my_list = []
    for x, y in zip(*indexes):
        my_list.append(np.where(flattened_pred_sorted == pred[x, y])[0][0])
    my_list.sort()
    print(my_list)




