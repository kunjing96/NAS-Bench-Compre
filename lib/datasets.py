import os
import json
import scipy.io as sio
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch
import logging


class Flowers(ImageFolder):
    def __init__(self, root, train=True, transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        label_path = os.path.join(root, 'imagelabels.mat')
        split_path = os.path.join(root, 'setid.mat')

        logging.warning('Dataset Flowers is trained with resolution 224!')

        # labels
        labels = sio.loadmat(label_path)['labels'][0]
        self.img_to_label = dict()
        for i in range(len(labels)):
            self.img_to_label[i] = labels[i]

        splits = sio.loadmat(split_path)
        self.trnid, self.valid, self.tstid = sorted(splits['trnid'][0].tolist()), \
                                             sorted(splits['valid'][0].tolist()), \
                                             sorted(splits['tstid'][0].tolist())
        if train:
            self.imgs = self.trnid + self.valid
        else:
            self.imgs = self.tstid

        self.samples = []
        for item in self.imgs:
            self.samples.append((os.path.join(root, 'jpg', "image_{:05d}.jpg".format(item)), self.img_to_label[item-1]-1))


class Cars196(ImageFolder, datasets.CIFAR10):
    base_folder_devkit = 'devkit'
    base_folder_trainims = 'cars_train'
    base_folder_testims = 'cars_test'

    filename_testanno = 'cars_test_annos.mat'
    filename_trainanno = 'cars_train_annos.mat'

    base_folder = 'cars_train'
    train_list = [
        ['00001.jpg', '8df595812fee3ca9a215e1ad4b0fb0c4'],
        ['00002.jpg', '4b9e5efcc3612378ec63a22f618b5028']
    ]
    test_list = []
    num_training_classes = 98 # 196/2

    def __init__(self, root, train=False, transform=None, target_transform=None, **kwargs):
        self.root = root
        self.transform = transform

        self.target_transform = target_transform
        self.loader = default_loader
        logging.warning('Dataset Cars196 is trained with resolution 224!')

        self.samples = []
        self.nb_classes = 196

        if train:
            labels = \
            sio.loadmat(os.path.join(self.root, self.base_folder_devkit, self.filename_trainanno))['annotations'][0]
            for item in labels:
                img_name = item[-1].tolist()[0]
                label = int(item[4]) - 1
                self.samples.append((os.path.join(self.root, self.base_folder_trainims, img_name), label))
        else:
            labels = \
            sio.loadmat(os.path.join(self.root, 'cars_test_annos_withlabels.mat'))['annotations'][0]
            for item in labels:
                img_name = item[-1].tolist()[0]
                label = int(item[-2]) - 1
                self.samples.append((os.path.join(self.root, self.base_folder_testims, img_name), label))


class Pets(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        train_list_path = os.path.join(self.dataset_root, 'annotations', 'trainval.txt')
        test_list_path = os.path.join(self.dataset_root, 'annotations', 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, 'images', "{}.jpg".format(img_name)), label-1))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, 'images', "{}.jpg".format(img_name)), label-1))


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def build_dataset(is_train, config, folder_name=None):
    transform = build_transform(is_train, config)

    if config.DATASET == 'CIFAR10':
        dataset = datasets.CIFAR10(config.DATAPATH, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif config.DATASET == 'CIFAR100':
        dataset = datasets.CIFAR100(config.DATAPATH, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif config.DATASET == 'CARS':
        dataset = Cars196(config.DATAPATH, train=is_train, transform=transform)
        nb_classes = 196
    elif config.DATASET == 'PETS':
        dataset = Pets(config.DATAPATH, train=is_train, transform=transform)
        nb_classes = 37
    elif config.DATASET == 'FLOWERS':
        dataset = Flowers(config.DATAPATH, train=is_train, transform=transform)
        nb_classes = 102
    elif config.DATASET == 'IMNET':
        root = os.path.join(config.DATAPATH, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATASET == 'EVO_IMNET':
        root = os.path.join(config.DATAPATH, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATASET == 'INAT':
        dataset = INatDataset(config.DATAPATH, train=is_train, year=2018,
                              category=config.INATCATEGORY, transform=transform)
        nb_classes = dataset.nb_classes
    elif config.DATASET == 'INAT19':
        dataset = INatDataset(config.DATAPATH, train=is_train, year=2019,
                              category=config.INATCATEGORY, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, config, img_size=None):
    if img_size is None:
        img_size = config.INPUTSIZE
    resize_im = img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=config.COLORJITTER,
            auto_augment=config.AA,
            interpolation=config.TRAIN_INTERPOLATION,
            re_prob=config.REPROB,
            re_mode=config.REMODE,
            re_count=config.RECOUNT,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def random_sample_valid_set(train_size, valid_size):
    assert train_size > valid_size

    g = torch.Generator()
    g.manual_seed(2147483647)  # set random seed before sampling validation set
    rand_indexes = torch.randperm(train_size, generator=g).tolist()

    valid_indexes = rand_indexes[:valid_size]
    train_indexes = rand_indexes[valid_size:]
    return train_indexes, valid_indexes