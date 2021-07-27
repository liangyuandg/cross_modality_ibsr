import numpy as np

TRAIN_X_PATH = '/yuanProject/XPath/ds96/train/X_train.npy'
TRAIN_Y_PATH = '/yuanProject/XPath/ds96/train/y_train.npy'
TEST_X_PATH = '/yuanProject/XPath/ds96/test/X_test.npy'
TEST_Y_PATH = '/yuanProject/XPath/ds96/test/y_test.npy'


data = np.load(TEST_X_PATH)
print(len(data), len(data[0]), len(data[1]), data[0][0].shape, data[0][1].shape)

data = np.load(TEST_Y_PATH)
print(data.shape, data)


data = np.load(TRAIN_X_PATH)
label = np.load(TRAIN_Y_PATH)
# mean, std of H&E
single_modal = data[0]
print(single_modal[label == 0].shape, single_modal[label == 1].shape, single_modal[label == 2].shape)
# mean for 1: [174.88023441 123.32407778 162.71974497] [46.81687903 49.13274691 38.91592081]
channels = single_modal[label == 1].reshape(-1, 3)
print(np.mean(channels, 0, keepdims=False), np.std(channels, 0, keepdims=False))
# mean for 0: [180.7491953  129.7742568  166.11690613] [46.76412872 59.06826753 44.51298519]
channels = single_modal[label == 0].reshape(-1, 3)
print(np.mean(channels, 0, keepdims=False), np.std(channels, 0, keepdims=False))
# mean for 2: [176.11142851 119.94298061 156.33634072] [49.60288757 57.74733246 48.50239955]
channels = single_modal[label == 2].reshape(-1, 3)
print(np.mean(channels, 0, keepdims=False), np.std(channels, 0, keepdims=False))
# total: [177.24695274 124.34710506 161.72433061] [47.7463798  55.49126494 44.1525292 ]
print(
    (np.mean(single_modal[label == 0].reshape(-1, 3), 0, keepdims=False) + np.mean(single_modal[label == 1].reshape(-1, 3), 0, keepdims=False) + np.mean(single_modal[label == 2].reshape(-1, 3), 0, keepdims=False)) / 3.0,
    np.sqrt((np.square(np.std(single_modal[label == 0].reshape(-1, 3), 0, keepdims=False)) + np.square(np.std(single_modal[label == 1].reshape(-1, 3), 0, keepdims=False)) + np.square(np.std(single_modal[label == 2].reshape(-1, 3), 0, keepdims=False))) / 3.0)
)


# mean, std of PPH3
single_modal = data[1]
print(single_modal[label == 0].shape, single_modal[label == 1].shape, single_modal[label == 2].shape)
# mean for 1: [174.88023441 123.32407778 162.71974497] [46.81687903 49.13274691 38.91592081]
channels = single_modal[label == 1].reshape(-1, 3)
print(np.mean(channels, 0, keepdims=False), np.std(channels, 0, keepdims=False))
# mean for 0: [180.7491953  129.7742568  166.11690613] [46.76412872 59.06826753 44.51298519]
channels = single_modal[label == 0].reshape(-1, 3)
print(np.mean(channels, 0, keepdims=False), np.std(channels, 0, keepdims=False))
# mean for 2: [176.11142851 119.94298061 156.33634072] [49.60288757 57.74733246 48.50239955]
channels = single_modal[label == 2].reshape(-1, 3)
print(np.mean(channels, 0, keepdims=False), np.std(channels, 0, keepdims=False))
# total: [194.24576912 186.99282001 200.96679032] [37.61597558 40.52679066 38.00598526]
print(
    (np.mean(single_modal[label == 0].reshape(-1, 3), 0, keepdims=False) + np.mean(single_modal[label == 1].reshape(-1, 3), 0, keepdims=False) + np.mean(single_modal[label == 2].reshape(-1, 3), 0, keepdims=False)) / 3.0,
    np.sqrt((np.square(np.std(single_modal[label == 0].reshape(-1, 3), 0, keepdims=False)) + np.square(np.std(single_modal[label == 1].reshape(-1, 3), 0, keepdims=False)) + np.square(np.std(single_modal[label == 2].reshape(-1, 3), 0, keepdims=False))) / 3.0)
)
