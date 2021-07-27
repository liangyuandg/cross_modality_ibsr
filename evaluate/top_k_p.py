from evaluate.miscs import draw_distance, draw_p_at_k
import numpy as np


# top K
K = [1, 5, 10]
QUERY_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_query_features.npy'
DATASET_FEATURE_PATH_1 = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_dataset_features.npy'
DATASET_FEATURE_PATH_2 = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/ph_dataset_features.npy'
QUERY_LABEL_PATH = '/yuanProject/XPath/ds96/test/y_test_balanced.npy'
DATASET_LABEL_PATH = '/yuanProject/XPath/ds96/train/y_train.npy'





def main():
    # read in data
    query_features = np.load(QUERY_FEATURE_PATH)
    dataset_features_1 = np.load(DATASET_FEATURE_PATH_1)
    dataset_features_2 = np.load(DATASET_FEATURE_PATH_2)
    query_gts = np.load(QUERY_LABEL_PATH)
    query_gts[query_gts == 2] = 0
    dataset_gts = np.load(DATASET_LABEL_PATH)
    dataset_gts[dataset_gts == 2] = 0

    for per_k in K:
        # overall
        overall_p = p_per_subset(query_features, dataset_features_1, dataset_features_2, dataset_gts, query_gts, per_k)
        # positive
        query_gts_subset = query_gts[query_gts == 1]
        query_features_subset = query_features[query_gts == 1]
        positive_p = p_per_subset(query_features_subset, dataset_features_1, dataset_features_2, dataset_gts, query_gts_subset, per_k)
        # negative
        query_gts_subset = query_gts[query_gts == 0]
        query_features_subset = query_features[query_gts == 0]
        negative_p = p_per_subset(query_features_subset, dataset_features_1, dataset_features_2, dataset_gts, query_gts_subset, per_k)

        print('{}: '.format(per_k), overall_p, positive_p, negative_p, '\n')


def p_per_subset(query_features, dataset_features_1, dataset_features_2, dataset_gts, query_gts, per_k):
    p_results = []
    # iterate for each image
    for case_numbering, item in enumerate(query_features):
        distances_1 = draw_distance(item, dataset_features_1)
        distances_2 = draw_distance(item, dataset_features_2)
        distances = np.minimum(distances_1, distances_2)

        # top k precision
        indexes = distances.argsort()[:per_k]
        best_match_label = [dataset_gts[index] for index in indexes]
        # print(best_match_label, query_gts[case_numbering])

        p = draw_p_at_k(query_gts[case_numbering], best_match_label)
        p_results.append(p)

    # summarize
    precision = sum(p_results) / len(p_results)

    return precision


if __name__ == '__main__':
    main()