import os
import argparse
import pickle
from PIL import Image

parser = argparse.ArgumentParser(description='Test script: loads and test a pre-trained model (e.g. after linear evaluation)')
parser.add_argument('--seed', default=-1, type=int, help='Seed for Numpy and pyTorch. Default: -1 (None)')
parser.add_argument('--backbone', default='resnet34', help='Backbone: conv4, resnet|8|32|34|56')
parser.add_argument('--data_size', default=32, type=int, help='Size of the mini-batch')
parser.add_argument("--num_workers", default=8, type=int, help="Number of torchvision workers used to load data (default: 8)")
parser.add_argument('--gpu', default="1", type=int, help='GPU id in case of multiple GPUs')
args = parser.parse_args()

CHECKPOINT = '/yuanProject/XPath/relationnet_ds96_512_two_tasks/epoch_151.tar'
TRAINING_DATA_PATH = '/yuanProject/XPath/ds96/train/X_train.npy'
TESTING_DATA_PATH = '/yuanProject/XPath/ds96/test/X_test_balanced.npy'
SAVE_PATH = '/yuanProject/XPath/relationnet_ds96_1024_two_tasks/reconstruction'
MODE = 'test'

HE_STAT = [[177.24695274/255.0, 124.34710506/255.0, 161.72433061/255.0], [47.7463798/255.0, 55.49126494/255.0, 44.1525292/255.0]]
PH_STAT = [[194.24576912/255.0, 186.99282001/255.0, 200.96679032/255.0], [37.61597558/255.0, 40.52679066/255.0, 38.00598526/255.0]]


import torch.optim
import numpy as np
import random
from models.relationnet import Model
from dataset.UnbalancedDataset_single_gpu import UnbalancedDataset
from backbones.decoder import ResNet_Decoder

def main():
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

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
        feature_decoder = ResNet_Decoder(inplane=256, layer_nums=[128, 64, 32, 16])
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

    # loading
    model.to(device)
    model.net.eval()
    feature_decoder.to(device)
    feature_decoder.eval()
    print("Loading checkpoint: " + str(CHECKPOINT))
    model.load(CHECKPOINT)
    print("Loading checkpoint: Done!")


    # training dataset
    # dataset
    dataset = UnbalancedDataset(
        data_path=TRAINING_DATA_PATH,
        label_path=None,
        repeat_augmentations=None,
        test_mode=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.data_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # run
    he_original_list = []
    ph_original_list = []
    he_reconstruct_list = []
    ph_reconstruct_list = []
    with torch.no_grad():
        for i, (he_image, ph_image) in enumerate(test_loader):
            he_original_list.append(he_image)
            ph_original_list.append(ph_image)
            data = torch.cat((he_image, ph_image), 0).to(device)
            inter_features, _ = model.test_forward(data)
            reconstruct = feature_decoder.test_forward(inter_features)

            he_reconstruct_list.append(reconstruct[0:reconstruct.shape[0]//2])
            ph_reconstruct_list.append(reconstruct[reconstruct.shape[0] // 2:])

    # save_to_image(
    #     torch.cat(he_original_list, dim=0).cpu().detach().numpy(),
    #     torch.cat(ph_original_list, dim=0).cpu().detach().numpy(),
    #     torch.cat(he_reconstruct_list, dim=0).cpu().detach().numpy(),
    #     torch.cat(ph_reconstruct_list, dim=0).cpu().detach().numpy(),
    #     'train'
    # )

    # testing dataset
    # dataset
    dataset = UnbalancedDataset(
        data_path=TESTING_DATA_PATH,
        label_path=None,
        repeat_augmentations=None,
        test_mode=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.data_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # run
    he_original_list = []
    ph_original_list = []
    he_reconstruct_list = []
    ph_reconstruct_list = []
    with torch.no_grad():
        for i, (he_image, ph_image) in enumerate(test_loader):
            he_original_list.append(he_image)
            ph_original_list.append(ph_image)
            data = torch.cat((he_image, ph_image), 0).to(device)
            inter_features, _ = model.test_forward(data)
            reconstruct = feature_decoder.test_forward(inter_features)

            he_reconstruct_list.append(reconstruct[0:reconstruct.shape[0]//2])
            ph_reconstruct_list.append(reconstruct[reconstruct.shape[0] // 2:])

    # save_to_image(
    #     torch.cat(he_original_list, dim=0).cpu().detach().numpy(),
    #     torch.cat(ph_original_list, dim=0).cpu().detach().numpy(),
    #     torch.cat(he_reconstruct_list, dim=0).cpu().detach().numpy(),
    #     torch.cat(ph_reconstruct_list, dim=0).cpu().detach().numpy(),
    #     'test'
    # )


# def save_to_image(list_1, list_2, list_3, list_4, mode):
#     import string
#     import random
#     for he_orginal, ph_original, he_reconstruct, ph_reconstruct in zip(list_1, list_2, list_3, list_4):
#         letters = string.ascii_lowercase
#         name = ''.join(random.choice(letters) for i in range(10))
#
#         tempx = np.transpose(he_orginal, (1, 2, 0))
#         tempx = ((tempx* HE_STAT[1] + HE_STAT[0])*255.0).astype(np.uint8)
#         Image.fromarray(tempx).save(os.path.join(SAVE_PATH, '{}_{}_he_original.jpg'.format(mode, name)))
#
#         tempx = np.transpose(ph_original, (1, 2, 0))
#         tempx = ((tempx* PH_STAT[1] + PH_STAT[0])*255.0).astype(np.uint8)
#         # tempx = (tempx / np.amax(tempx) * 255).astype(np.uint8)
#         Image.fromarray(tempx).save(os.path.join(SAVE_PATH, '{}_{}_ph_original.jpg'.format(mode, name)))
#
#         tempx = np.transpose(he_reconstruct, (1, 2, 0))
#         tempx = ((tempx * HE_STAT[1] + HE_STAT[0])*255.0).astype(np.uint8)
#         Image.fromarray(tempx).save(os.path.join(SAVE_PATH, '{}_{}_he_reconstruct.jpg'.format(mode, name)))
#
#         tempx = np.transpose(ph_reconstruct, (1, 2, 0))
#         tempx = ((tempx* PH_STAT[1] + PH_STAT[0])*255.0).astype(np.uint8)
#         Image.fromarray(tempx).save(os.path.join(SAVE_PATH, '{}_{}_ph_reconstruct.jpg'.format(mode, name)))


if __name__== "__main__": main()