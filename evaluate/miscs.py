import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def draw_distance(query_feature, feature_list):
    distance = []
    for item in feature_list:
        dis = np.linalg.norm(query_feature - item)
        distance.append(dis)
    distance = np.asarray(distance)

    return distance


def classification_eval(gts, predictions):
    acc = accuracy_score(gts, predictions)
    precision = precision_score(gts, predictions, average=None)
    recall = recall_score(gts, predictions, average=None)
    f1 = f1_score(gts, predictions, average=None)

    return acc, precision, recall, f1


def draw_p_at_k(gt, predictions):
    p = float(np.count_nonzero(predictions == gt)) / float(len(predictions))

    return p


def draw_ap(gt, predictions):
    ap = 0.0
    running_positives = 0
    for idx, i in enumerate(predictions):
        if i == gt:
            running_positives += 1
            ap_at_count = running_positives/(idx+1)
            ap += ap_at_count
    return ap/np.sum(gt==predictions)
