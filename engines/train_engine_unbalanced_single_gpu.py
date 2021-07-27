import os
import argparse
import time
from miscs.visdom_utils import VisdomPlotter

import torch.optim
import numpy as np
import random
from models.relationnet import Model
from dataset.UnbalancedDataset_single_gpu import UnbalancedDataset
from torch.optim import SGD, Adam
from utils import AverageMeter
from evaluate.top_k_ap import map_per_subset

parser = argparse.ArgumentParser(description="Training script for the unsupervised phase via self-supervision")
parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
parser.add_argument("--epochs", default=400, type=int, help="Total number of epochs")
parser.add_argument("--backbone", default="resnet34", help="Backbone: conv4, resnet|8|32|34|56")
parser.add_argument("--method", default="relationnet", help="Model: standard, randomweights, relationnet, rotationnet, deepinfomax, simclr")
parser.add_argument("--data_size", default=32, type=int, help="Size of the mini-batch")
parser.add_argument("--K", default=8, type=int, help="Total number of augmentations (K), sed only in RelationNet")
parser.add_argument("--aggregation", default="cat", help="Aggregation function used in RelationNet: sum, mean, max, cat")
parser.add_argument("--num_workers", default=8, type=int, help="Number of torchvision workers used to load data (default: 8)")
parser.add_argument("--gpu", default="1", type=str, help="GPU id in case of multiple GPUs")
args = parser.parse_args()

DATA_PATH = '/yuanProject/XPath/ds96/train/X_train.npy'
LABEL_PATH = '/yuanProject/XPath/ds96/train/y_train.npy'

DATA_PATH_REPO = '/yuanProject/XPath/ds96/train/X_train.npy'
DATA_PATH_QUERY = '/yuanProject/XPath/ds96/test/X_test_balanced.npy'
LABEL_PATH_REPO = '/yuanProject/XPath/ds96/train/y_train.npy'
LABEL_PATH_QUERY = '/yuanProject/XPath/ds96/test/y_test_balanced.npy'

WORKSPACE = '/yuanProject/XPath/relationnet_ds96_512_continue_training_adam'
MODE = 'train'
# CHECKPOINT = '/yuanProject/XPath/pretrained/relationnet_slim_resnet34_seed_1_epoch_300.tar'
# CHECKPOINT = None
# CHECKPOINT = '/yuanProject/XPath/relationnet_ds96_1024/epoch_200.tar'
SAVE_EPOCH = 1
VALIDATE_EPOCH = 1


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if (args.seed >= 0):
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

    if (torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    if (args.backbone == "resnet8"):
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

    tot_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    print("[INFO]", str(str(args.backbone)), "loaded in memory.")
    print("[INFO] Feature size:", str(feature_extractor.feature_size))
    print("[INFO] Feature extractor TOT trainable params: " + str(tot_params))
    print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type: " + str(device))

    model = Model(feature_extractor, mode=MODE, device=device, aggregation=args.aggregation)

    # training dataset
    train_dataset = UnbalancedDataset(
        data_path=DATA_PATH,
        label_path=LABEL_PATH,
        repeat_augmentations=args.K,
        test_mode=False,
    )
    print("[INFO][RelationNet] TOT augmentations (K): " + str(args.K))
    print("[INFO][RelationNet] Aggregation function: " + str(args.aggregation))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.data_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # testing dataset for repo
    test_dataset_repo = UnbalancedDataset(
        data_path=DATA_PATH_REPO,
        label_path=None,
        repeat_augmentations=None,
        test_mode=True,
    )
    test_loader_repo = torch.utils.data.DataLoader(
        test_dataset_repo, batch_size=args.data_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # testing dataset for query
    test_dataset_query = UnbalancedDataset(
        data_path=DATA_PATH_QUERY,
        label_path=None,
        repeat_augmentations=None,
        test_mode=True,
    )
    test_loader_query = torch.utils.data.DataLoader(
        test_dataset_query, batch_size=args.data_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # logging
    if not os.path.exists(WORKSPACE):
        os.makedirs(WORKSPACE)
    log_file = os.path.join(WORKSPACE, "log.cvs")
    with open(log_file, "w") as myfile:
        myfile.write("epoch,loss,score" + "\n") # create a new log file (it destroys the previous one)

    # loading
    model.to(device)
    if CHECKPOINT != None:
        print("Loading checkpoint: " + str(CHECKPOINT))
        model.load(CHECKPOINT)
        print("Loading checkpoint: Done!")

    # optimizer
    optimizer = Adam([{"params": model.net.parameters(), "lr": 0.001, "weight_decay": 5e-4},
                           {"params": model.relation_module.parameters(), "lr": 0.001, "weight_decay": 5e-4}])
    # optimizer = SGD([{"params": model.net.parameters(), "lr": 0.001, "momentum": 0.9},
    #                        {"params": model.relation_module.parameters(), "lr": 0.001, "momentum": 0.9}])

    model.net.train()
    model.relation_module.train()
    for epoch in range(args.epoch_start, args.epochs):

        print('[INFO] starting training epoch: {}'.format(epoch))
        # training
        model.net.train()
        model.relation_module.train()
        start_time = time.time()
        accuracy_pos_list = list()
        accuracy_neg_list = list()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        statistics_dict = {}
        for i, (he_image, ph_image, he_image_list, ph_image_list, _) in enumerate(train_loader):
            batch_size = he_image.shape[0]
            tot_augmentations = len(he_image_list)
            train_x_he = torch.cat(he_image_list, 0)
            train_x_ph = torch.cat(ph_image_list, 0)
            train_x = torch.cat((train_x_he, train_x_ph), 0).to(device)

            _, predictions, loss, train_y, tot_pairs, tot_positive, tot_negative = model.train_forward(train_x, tot_augmentations, batch_size)

            optimizer.zero_grad()
            loss_meter.update(loss.item(), len(train_y))
            # backward step and weights update
            loss.backward()
            optimizer.step()

            best_guess = torch.round(torch.sigmoid(predictions))
            correct = best_guess.eq(train_y.view_as(best_guess))
            correct_positive = correct[0:int(len(correct) / 2)].cpu().sum()
            correct_negative = correct[int(len(correct) / 2):].cpu().sum()
            correct = correct.cpu().sum()
            accuracy = (100.0 * correct / float(len(train_y)))
            accuracy_meter.update(accuracy.item(), len(train_y))
            accuracy_pos_list.append((100.0 * correct_positive / float(len(train_y) / 2)).item())
            accuracy_neg_list.append((100.0 * correct_negative / float(len(train_y) / 2)).item())

            if (i == 0):
                statistics_dict["batch_size"] = batch_size
                statistics_dict["tot_pairs"] = tot_pairs
                statistics_dict["tot_positive"] = int(tot_positive)
                statistics_dict["tot_negative"] = int(tot_negative)

        # epoch stat
        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]"
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg)
              + "; acc: " + str(accuracy_meter.avg) + "%"
              + "; acc+: " + str(round(np.mean(accuracy_pos_list), 2)) + "%"
              + "; acc-: " + str(round(np.mean(accuracy_neg_list), 2)) + "%"
              + "; batch-size: " + str(statistics_dict["batch_size"])
              + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))

        with open(log_file, "a") as myfile:
                myfile.write(str(epoch)+","+str(loss_meter.avg)+","+str(accuracy_meter.avg)+"\n")

        # draw to visdom
        plotter.line_plot(var_name='loss', split_name='train', title_name='Loss', x=epoch, y=loss_meter.avg)
        plotter.line_plot(var_name='acc', split_name='acc', title_name='Acc', x=epoch, y=accuracy_meter.avg)
        plotter.line_plot(var_name='acc', split_name='acc+', title_name='Acc', x=epoch, y=round(np.mean(accuracy_pos_list), 2))
        plotter.line_plot(var_name='acc', split_name='acc-', title_name='Acc', x=epoch, y=round(np.mean(accuracy_neg_list), 2))

        # validate model
        if (epoch+1) % VALIDATE_EPOCH == 0:
            print('[INFO] starting validation epoch: {}'.format(epoch))
            query_he_embeddings_list = []
            dataset_he_embeddings_list = []
            dataset_ph_embeddings_list = []
            model.net.eval()
            model.relation_module.eval()
            with torch.no_grad():
                for i, (he_image, ph_image) in enumerate(test_loader_query):
                    data = torch.cat((he_image, ph_image), 0).to(device)
                    _, output = model.test_forward(data)
                    query_he_embeddings_list.append(output[0:output.shape[0] // 2])
                for i, (he_image, ph_image) in enumerate(test_loader_repo):
                    data = torch.cat((he_image, ph_image), 0).to(device)
                    _, output = model.test_forward(data)
                    dataset_he_embeddings_list.append(output[0:output.shape[0] // 2])
                    dataset_ph_embeddings_list.append(output[output.shape[0] // 2:])

            query_he_embeddings_list = torch.cat(query_he_embeddings_list, dim=0).cpu().detach().numpy()
            dataset_he_embeddings_list = torch.cat(dataset_he_embeddings_list, dim=0).cpu().detach().numpy()
            dataset_ph_embeddings_list = torch.cat(dataset_ph_embeddings_list, dim=0).cpu().detach().numpy()

            query_gts = np.load(LABEL_PATH_QUERY)
            query_gts[query_gts == 2] = 0
            dataset_gts = np.load(LABEL_PATH_REPO)
            dataset_gts[dataset_gts == 2] = 0

            # overall
            overall_map = map_per_subset(query_he_embeddings_list, dataset_he_embeddings_list, dataset_ph_embeddings_list, dataset_gts, query_gts)
            # positive
            query_gts_subset = query_gts[query_gts == 1]
            query_features_subset = query_he_embeddings_list[query_gts == 1]
            positive_map = map_per_subset(query_features_subset, dataset_he_embeddings_list, dataset_ph_embeddings_list, dataset_gts, query_gts_subset)
            # negative
            query_gts_subset = query_gts[query_gts == 0]
            query_features_subset = query_he_embeddings_list[query_gts == 0]
            negative_map = map_per_subset(query_features_subset, dataset_he_embeddings_list, dataset_ph_embeddings_list, dataset_gts, query_gts_subset)

            print('[INFO] validation acc: {}, pos acc: {}, neg acc: {}'.format(overall_map, positive_map, negative_map))

            # draw to visdom
            plotter.line_plot(var_name='topk', split_name='topk', title_name='TopK', x=epoch, y=overall_map)
            plotter.line_plot(var_name='topk', split_name='topk+', title_name='TopK', x=epoch, y=positive_map)
            plotter.line_plot(var_name='topk', split_name='topk-', title_name='TopK', x=epoch, y=negative_map)

        # save model
        if (epoch+1) % SAVE_EPOCH == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(WORKSPACE, "epoch_{}.tar".format(str(epoch+1)))
            print("[INFO] Saving in:", checkpoint_path)
            model.save(checkpoint_path)


if __name__== "__main__":
    global plotter
    plotter = VisdomPlotter(env_name='XPath')
    main()
