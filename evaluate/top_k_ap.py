from evaluate.miscs import draw_distance, draw_ap
import numpy as np


QUERY_FEATURE_PATH = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_query_features.npy'
DATASET_FEATURE_PATH_1 = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/he_dataset_features.npy'
DATASET_FEATURE_PATH_2 = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/ph_dataset_features.npy'
QUERY_LABEL_PATH = '/yuanProject/XPath/ds96/test/y_test_balanced.npy'
DATASET_LABEL_PATH = '/yuanProject/XPath/ds96/train/y_train.npy'

# two task epoch 153  0.6220375167605705 0.3296208323946227 0.9144542011265194
# one task epoch 31 0.6377679093299529 0.3516632086969764 0.9238726099629303

def main():

    # read in data
    query_features = np.load(QUERY_FEATURE_PATH)
    dataset_features_1 = np.load(DATASET_FEATURE_PATH_1)
    dataset_features_2 = np.load(DATASET_FEATURE_PATH_2)
    query_gts = np.load(QUERY_LABEL_PATH)
    query_gts[query_gts == 2] = 0
    dataset_gts = np.load(DATASET_LABEL_PATH)
    dataset_gts[dataset_gts == 2] = 0

    # overall
    overall_map = map_per_subset(query_features, dataset_features_1, dataset_features_2, dataset_gts, query_gts)
    # positive
    query_gts_subset = query_gts[query_gts == 1]
    query_features_subset = query_features[query_gts == 1]
    positive_map = map_per_subset(query_features_subset, dataset_features_1, dataset_features_2, dataset_gts, query_gts_subset)
    # negative
    query_gts_subset = query_gts[query_gts == 0]
    query_features_subset = query_features[query_gts == 0]
    negative_map = map_per_subset(query_features_subset, dataset_features_1, dataset_features_2, dataset_gts, query_gts_subset)

    print(overall_map, positive_map, negative_map)



def map_per_subset(query_features, dataset_features_1, dataset_features_2, dataset_gts, query_gts):
    ap_results = []
    # iterate for each image
    for case_numbering, item in enumerate(query_features):
        distances_1 = draw_distance(item, dataset_features_1)
        distances_2 = draw_distance(item, dataset_features_2)
        distances = np.minimum(distances_1, distances_2)

        # top k map
        indexes = distances.argsort()
        all_match_label = [dataset_gts[index] for index in indexes]
        ap = draw_ap(query_gts[case_numbering], all_match_label)
        ap_results.append(ap)

    # summarize
    meanap = sum(ap_results) / len(ap_results)

    return meanap



if __name__ == '__main__':
    main()