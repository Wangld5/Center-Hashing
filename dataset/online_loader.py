import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm


def config_dataset(config):
    if 'CUB' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 200
    elif 'nabirds' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 555
    elif 'car' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 196
    # elif 'imagenet' in config['dataset']:
    #     config['topk'] = 1000
    #     config['n_class'] = 100
    elif 'coco' in config['dataset']:
        config['topk'] = 5000
        config['n_class'] = 80
    elif 'imagenet1k' in config['dataset']:
        config['topk'] = 1000
        config['n_class'] = 1000


    if config['dataset'] == 'CUB_200_2011':
        config['data_path'] = '../../data/' + config['dataset'] + '/images/'
    if config['dataset'] == 'nabirds':
        config['data_path'] = '../../data/' + config['dataset'] + '/images/'
    if config['dataset'] == 'car_ims':
        config['data_path'] = '../../data/'
    if config['dataset'] == 'imagenet':
        config['data_path'] = '../../data/imagenet/'
    if config['dataset'] == 'mscoco':
        config['data_path'] = '../../data/coco/'
    if config['dataset'] == 'imagenet1k':
        config['data_path'] = '../../data/'
    
    if config['dataset'] == 'imagenet' or config['dataset'] == 'mscoco' or config['dataset'] == 'imagenet1k':
        config['data'] = {
            'train_set': {'list_path': './data/' + config['dataset'] + '/train.txt', 'batch_size': config['batch_size']},
            'test_set': {'list_path': './data/' + config['dataset'] + '/test.txt', 'batch_size': config['batch_size']},
            'database': {'list_path': './data/' + config['dataset'] + '/database.txt', 'batch_size': config['batch_size']},
        }
    else:
        config['data'] = {
            'train_set': {'list_path': './data/' + config['dataset'] + '/train.txt', 'batch_size': config['batch_size']},
            'test_set': {'list_path': './data/' + config['dataset'] + '/test.txt', 'batch_size': config['batch_size']},
            'database': {'list_path': './data/' + config['dataset'] + '/train.txt', 'batch_size': config['batch_size']},
        }

    return config


def encode_onehot(labels, num_classes=10):
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


class ImageList(object):
    def __init__(self, data_path, image_list, transform, n_class):
        self.imgs = [data_path + val.strip().split('\t')[0] for val in image_list]
        self.transform = transform
        self.labels = [int(val.strip().split('\t')[1]) for val in image_list]
        self.n_class = n_class

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        target = np.eye(self.n_class, dtype=np.float32)[np.array(target)]
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def get_one_hot_label(self):
        return torch.from_numpy(encode_onehot(self.labels, self.n_class)).float()

class ImageNet(object):
    def __init__(self, data_path, image_list, transform, n_class):
        self.imgs = [(data_path + val.strip().split()[0], np.array([float(la) for la in val.strip().split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        target = target.astype(np.float32)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config['data']

    for data_set in ['train_set', 'test_set', 'database']:
        dsets[data_set] = ImageList(config['data_path'],
                                    open(data_config[data_set]['list_path']).readlines(),
                                    transform=image_transform(config['resize_size'], config['crop_size'], data_set),
                                    n_class=config['n_class'])
        print(data_set, len(dsets[data_set]))
        if data_set == 'train_set':
            isShuffle = True
        else:
            isShuffle = False
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]['batch_size'],
                                                      shuffle=isShuffle, num_workers=4)
    return dset_loaders['train_set'], dset_loaders['test_set'], dset_loaders['database'], \
           len(dsets['train_set']), len(dsets['test_set']), len(dsets['database'])

def get_imagenet_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test_set", "database"]:
        dsets[data_set] = ImageNet(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set),
                                    n_class=config['n_class'])
        print(data_set, len(dsets[data_set]))
        if data_set == 'train_set':
            isShuffle = True
        else:
            isShuffle = False
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=isShuffle, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test_set"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test_set"]), len(dsets["database"])


if __name__ == '__main__':
    config = get_config()
    train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(config)
    for img, label, ind in train_loader:
        print(img.shape)
        print(label.shape)
        break
