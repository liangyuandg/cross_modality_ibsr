import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='Test script: loads and test a pre-trained model (e.g. after linear evaluation)')
parser.add_argument('--seed', default=-1, type=int, help='Seed for Numpy and pyTorch. Default: -1 (None)')
parser.add_argument('--backbone', default='resnet34', help='Backbone: conv4, resnet|8|32|34|56')
parser.add_argument('--data_size', default=32, type=int, help='Size of the mini-batch')
parser.add_argument("--num_workers", default=8, type=int, help="Number of torchvision workers used to load data (default: 8)")
parser.add_argument('--gpu', default="1", type=int, help='GPU id in case of multiple GPUs')
args = parser.parse_args()

CHECKPOINT = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/epoch_31.tar'
DATA_PATH = '/yuanProject/XPath/ds96/test/X_test_balanced.npy'
LABEL_PATH = '/yuanProject/XPath/ds96/test/y_test_balanced.npy'
# DATA_PATH = '/yuanProject/XPath/ds96/train/X_train.npy'
# LABEL_PATH = '/yuanProject/XPath/ds96/train/y_train.npy'
SAVE_PATH = '/yuanProject/XPath/relationnet_ds96_1024_continue_training_adam/'
MODE = 'test'

import torch.optim
import numpy as np
import random
from models.relationnet import Model
from dataset.UnbalancedDataset_single_gpu import UnbalancedDataset

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if(args.seed>=0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[INFO] Setting SEED: " + str(args.seed))
    else:
        print("[INFO] Setting SEED: None")

    if(torch.cuda.is_available() == False): print('[WARNING] CUDA is not available.')

    if (args.backbone == "conv4"):
        from backbones.conv4 import Conv4
        feature_extractor = Conv4(flatten=True)
    elif (args.backbone == "resnet8"):
        from backbones.resnet_small import ResNet, BasicBlock
        feature_extractor = ResNet(BasicBlock, [1, 1, 1], channels=[16, 32, 64], flatten=True)
    elif (args.backbone == "resnet32"):
        from backbones.resnet_small import ResNet, BasicBlock
        feature_extractor = ResNet(BasicBlock, [5, 5, 5], channels=[16, 32, 64], flatten=True)
    elif (args.backbone == "resnet56"):
        from backbones.resnet_small import ResNet, BasicBlock
        feature_extractor = ResNet(BasicBlock, [9, 9, 9], channels=[16, 32, 64], flatten=True)
    elif (args.backbone == "resnet34"):
        from backbones.resnet_large import ResNet, BasicBlock
        feature_extractor = ResNet(BasicBlock, layers=[3, 4, 6, 3], zero_init_residual=False,
                                   groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                   norm_layer=None)
    elif (args.backbone == "resnet10"):
        from backbones.resnet_large import ResNet, BasicBlock
        feature_extractor = ResNet(BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                                   groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                   norm_layer=None)
    else:
        raise RuntimeError("[ERROR] the backbone " + str(args.backbone) + " is not supported.")

    print("[INFO]", str(str(args.backbone)), "loaded in memory.")
    print("[INFO] Feature size:", str(feature_extractor.feature_size))

    print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type: " + str(device))

    # modeling
    model = Model(feature_extractor, mode=MODE, device=device, aggregation=None)

    # dataset
    dataset = UnbalancedDataset(
        data_path=DATA_PATH,
        label_path=LABEL_PATH,
        repeat_augmentations=None,
        test_mode=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.data_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # loading
    model.to(device)
    model.net.eval()
    print("Loading checkpoint: " + str(CHECKPOINT))
    model.load(CHECKPOINT)
    print("Loading checkpoint: Done!")

    # run
    he_embeddings_list = []
    ph_embeddings_list = []
    with torch.no_grad():
        for i, (he_image, ph_image) in enumerate(test_loader):
            data = torch.cat((he_image, ph_image), 0).to(device)
            _, output = model.test_forward(data)
            he_embeddings_list.append(output[0:output.shape[0]//2])
            ph_embeddings_list.append(output[output.shape[0] // 2:])

    with open(os.path.join(SAVE_PATH, 'he_query_features.npy'), 'wb') as handle:
        np.save(handle, torch.cat(he_embeddings_list, dim=0).cpu().detach().numpy())

    with open(os.path.join(SAVE_PATH, 'ph_query_features.npy'), 'wb') as handle:
        np.save(handle, torch.cat(ph_embeddings_list, dim=0).cpu().detach().numpy())


if __name__== "__main__": main()