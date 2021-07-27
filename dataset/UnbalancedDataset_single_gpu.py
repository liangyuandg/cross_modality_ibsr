from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import string
import random

HE_STAT = [[177.24695274/255.0, 124.34710506/255.0, 161.72433061/255.0], [47.7463798/255.0, 55.49126494/255.0, 44.1525292/255.0]]
PH_STAT = [[194.24576912/255.0, 186.99282001/255.0, 200.96679032/255.0], [37.61597558/255.0, 40.52679066/255.0, 38.00598526/255.0]]


class UnbalancedDataset(Dataset):
    """dental dataset for detection.
    """

    def __init__(
            self,
            data_path,
            label_path,
            repeat_augmentations,
            test_mode=False,
    ):
        """Dataset class.

        Args:
            data_path.
            label_path.
            test_mode (bool): test mode.

        Returns:

        """
        self.test_mode = test_mode
        self.repeat_augmentations = repeat_augmentations
        self.he_data = np.load(data_path)[0]
        self.ph_data = np.load(data_path)[1]
        print(len(self.he_data))
        if label_path is not None:
            self.labels = np.load(label_path)

        self.train_he_transform = self.get_train_he_transforms()
        self.train_ph_transform = self.get_train_ph_transforms()
        self.test_he_transform = self.get_test_he_transforms()
        self.test_ph_transform = self.get_test_ph_transforms()


    def __len__(self):
        return len(self.he_data)

    def _rand_another(self):
        return np.random.choice(list(range(0, len(self))))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def get_train_he_transforms(self):
        normalize = transforms.Normalize(mean=[177.24695274/255.0, 124.34710506/255.0, 161.72433061/255.0], std=[47.7463798/255.0, 55.49126494/255.0, 44.1525292/255.0])
        side = 96
        padding = 4
        cutout = 0.0625

        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue=0.1)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        # rnd_gray = transforms.RandomGrayscale(p=0.2)
        rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333), interpolation=2)
        rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
        rnd_vflip = transforms.RandomVerticalFlip(p=0.5)
        # rnd_rot = transforms.RandomRotation(10., resample=2)
        rnd_trans = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=None, shear=None, resample=2, fillcolor=0)
        # train_transform = transforms.Compose(
        #     [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_rot, rnd_color_jitter, transforms.ToTensor(), normalize]
        # )
        # train_transform = transforms.Compose(
        #     [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_rot, rnd_color_jitter, transforms.ToTensor()]
        # )
        train_transform = transforms.Compose(
            [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_color_jitter, rnd_trans, transforms.ToTensor(), normalize]
        )
        # train_transform = transforms.Compose(
        #     [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_color_jitter, rnd_trans, transforms.ToTensor()]
        # )


        return train_transform

    def get_train_ph_transforms(self):
        normalize = transforms.Normalize(mean=[194.24576912/255.0, 186.99282001/255.0, 200.96679032/255.0], std=[37.61597558/255.0, 40.52679066/255.0, 38.00598526/255.0])
        side = 96
        # padding = 4
        # cutout = 0.0625

        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue=0.1)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        # rnd_gray = transforms.RandomGrayscale(p=0.2)
        rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333), interpolation=2)
        rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
        rnd_vflip = transforms.RandomVerticalFlip(p=0.5)
        # rnd_rot = transforms.RandomRotation(10., resample=2)
        rnd_trans = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=None, shear=None, resample=2, fillcolor=0)
        # train_transform = transforms.Compose(
        #     [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_rot, rnd_color_jitter, transforms.ToTensor(), normalize]
        # )
        # train_transform = transforms.Compose(
        #     [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_rot, rnd_color_jitter, transforms.ToTensor()]
        # )
        train_transform = transforms.Compose(
            [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_color_jitter, rnd_trans, transforms.ToTensor(), normalize]
        )
        # train_transform = transforms.Compose(
        #     [rnd_resizedcrop, rnd_hflip, rnd_vflip, rnd_color_jitter, rnd_trans, transforms.ToTensor()]
        # )


        return train_transform

    def get_test_he_transforms(self):
        normalize = transforms.Normalize(mean=[177.24695274/255.0, 124.34710506/255.0, 161.72433061/255.0], std=[47.7463798/255.0, 55.49126494/255.0, 44.1525292/255.0])
        # normalize = transforms.Normalize(mean=[177.24695274, 124.34710506, 161.72433061], std=[47.7463798, 55.49126494, 44.1525292])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        # test_transform = transforms.Compose([transforms.ToTensor()])

        return test_transform

    def get_test_ph_transforms(self):
        normalize = transforms.Normalize(mean=[194.24576912/255.0, 186.99282001/255.0, 200.96679032/255.0], std=[37.61597558/255.0, 40.52679066/255.0, 38.00598526/255.0])
        # normalize = transforms.Normalize(mean=[194.24576912, 186.99282001, 200.96679032], std=[37.61597558, 40.52679066, 38.00598526])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        # test_transform = transforms.Compose([transforms.ToTensor()])

        return test_transform

    def prepare_train_img(self, idx):
        he_image, ph_image, target = self.he_data[idx], self.ph_data[idx], int(self.labels[idx])

        # original image
        # temp_data_1 = np.copy(he_image)
        # temp_data_2 = np.copy(ph_image)
        # letters = string.ascii_lowercase
        # name = ''.join(random.choice(letters) for i in range(10))
        # work_dir = '/yuanProject/XPath/relationnet_ds96_512_two_task_adam/original/'
        # Image.fromarray(temp_data_1).save(work_dir + '{}_1.jpg'.format(name))
        # Image.fromarray(temp_data_2).save(work_dir + '{}_2.jpg'.format(name))

        he_pic = Image.fromarray(he_image.astype(np.uint8))
        ph_pic = Image.fromarray(ph_image.astype(np.uint8))

        he_image_list = list()
        ph_image_list = list()
        count = 0

        for _ in range(self.repeat_augmentations):
            # tempx = np.transpose(self.train_transform(ph_pic.copy()).cpu().detach().numpy(), (1, 2, 0))
            # tempx = (tempx / np.amax(tempx) * 255).astype(np.uint8)
            # # original image
            # work_dir = '/home/yuan/self-supervised-relational-reasoning/temp/'
            # Image.fromarray(tempx).save(work_dir + '{}.jpg'.format(name+'{}'.format(i)))
            # import torch
            # print(torch.std(self.train_he_transform(he_pic.copy())), torch.mean(self.train_he_transform(he_pic.copy())), torch.std(self.train_ph_transform(ph_pic.copy())), torch.mean(self.train_ph_transform(ph_pic.copy())))

            # temp_data_1 = self.train_he_transform(he_pic.copy())
            # temp_data_2 = self.train_ph_transform(ph_pic.copy())
            # letters = string.ascii_lowercase
            # name = ''.join(random.choice(letters) for i in range(10))
            # work_dir = '/yuanProject/XPath/relationnet_ds96_512_two_task_adam/augmented/'
            # Image.fromarray(((temp_data_1.detach().cpu().numpy() * np.asarray(HE_STAT[1])[:, None, None] + np.asarray(HE_STAT[0])[:, None, None]) * 255.0).astype(np.uint8).transpose((1, 2, 0))).save(work_dir + '{}_{}_1.jpg'.format(name, count))
            # Image.fromarray(((temp_data_2.detach().cpu().numpy() * np.asarray(PH_STAT[1])[:, None, None] + np.asarray(PH_STAT[0])[:, None, None]) * 255.0).astype(np.uint8).transpose((1, 2, 0))).save(work_dir + '{}_{}_2.jpg'.format(name, count))
            # he_image_list.append(temp_data_1)
            # ph_image_list.append(temp_data_2)
            # count = count + 1

            he_image_list.append(self.train_he_transform(he_pic.copy()))
            ph_image_list.append(self.train_ph_transform(ph_pic.copy()))

        return he_image, ph_image, he_image_list, ph_image_list, target


    def prepare_test_img(self, idx):
        he_image, ph_image = self.he_data[idx], self.ph_data[idx]
        he_pic = Image.fromarray(he_image.astype(np.uint8))
        ph_pic = Image.fromarray(ph_image.astype(np.uint8))

        he_image = self.test_he_transform(he_pic.copy())
        ph_image = self.test_ph_transform(ph_pic.copy())

        return he_image, ph_image


