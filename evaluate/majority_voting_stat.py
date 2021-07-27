from evaluate.miscs import draw_distance, classification_eval
import numpy as np

# top K
K = 5
QUERY_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_query_features.npy'
DATASET_FEATURE_PATH_1 = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_dataset_features.npy'
DATASET_FEATURE_PATH_2 = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/ph_dataset_features.npy'
QUERY_LABEL_PATH = '/yuanProject/XPath/ds96/test/y_test_balanced.npy'
DATASET_LABEL_PATH = '/yuanProject/XPath/ds96/train/y_train.npy'

# 0.7654028436018957 [0.97630332 0.55450237] [0.68666667 0.95901639] [0.80626223 0.7027027 ]
def main():
    # read in data
    query_features = np.load(QUERY_FEATURE_PATH)
    dataset_features = np.load(DATASET_FEATURE_PATH_1)
    if DATASET_FEATURE_PATH_2 is not None:
        dataset_features = np.concatenate((dataset_features, np.load(DATASET_FEATURE_PATH_2)), axis=0)
    query_gts = np.load(QUERY_LABEL_PATH)
    query_gts[query_gts == 2] = 0
    dataset_gts = np.load(DATASET_LABEL_PATH)
    if DATASET_FEATURE_PATH_2 is not None:
        dataset_gts = np.concatenate((dataset_gts, dataset_gts), axis=0)
    dataset_gts[dataset_gts == 2] = 0

    results = []
    # iterate for each image
    for item in query_features:
        distances = draw_distance(item, dataset_features)
        indexes = distances.argsort()[:K]
        best_match_label = [dataset_gts[index] for index in indexes]
        # predicted_label = np.bincount(best_match_label).argmax()
        predicted_label = max(set(best_match_label), key = best_match_label.count)
        results.append(predicted_label)
        # best_match_value = [distances[index] for index in indexes]
        # print(best_match_value)
    # summarize
    acc, precision, recall, f1 = classification_eval(gts=query_gts, predictions=results)
    print(acc, precision, recall, f1)


if __name__ == '__main__':
    main()