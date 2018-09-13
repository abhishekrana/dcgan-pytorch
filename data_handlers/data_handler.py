import os
import numpy as np
import torch
from torchvision import datasets, transforms
import logging
import glob
import pudb


# class DataHandler(torch.utils.data.Dataset):
class DataHandler():

    def __init__(self, config, mode='TRAIN'):
        # super(Cifar10DataLoader, self).__init__(config)
        self.config = config
        self.data_dir = config['data_handler']['data_dir']
        self.data_name = config['data_handler']['data_name']
        self.batch_size = config['data_handler']['batch_size']
        self.shuffle = config['data_handler']['shuffle']
        # self.dataset_split = ['train', 'val', 'test']
        self.dataset_split = ['train', 'test']


        # https://github.com/uploadcare/pillow-simd

        # IMG_SIZE = 224
        # IMG_SIZE = 32
        IMG_SIZE = 64
        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]

        # Torchvision reads datasets into PILImage (Python imaging format).
        # ToTensor converts the PIL Image from range [0, 255] to a FloatTensor of shape (C x H x W) with range [0.0, 1.0]
        data_transforms = {
            'train': transforms.Compose([
               transforms.Resize(IMG_SIZE),
               transforms.CenterCrop(IMG_SIZE),
               # transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            # 'val': transforms.Compose([
            #    transforms.Resize(IMG_SIZE),
            #    transforms.CenterCrop(IMG_SIZE),
            #    transforms.ToTensor(),
            #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # ]),
            'test': transforms.Compose([
               transforms.Resize(IMG_SIZE),
               transforms.CenterCrop(IMG_SIZE),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),

        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, self.data_name, x),
                                                  data_transforms[x])
                          for x in self.dataset_split}


        self.dataloaders = {
            'train' : torch.utils.data.DataLoader(
                        image_datasets['train'], 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        pin_memory=True,                                       # To speed up RAM to GPU transfers (and does nothing for CPU-only code)
                                                                               # pinned memory is page-locked memory. It is easy for users to shoot themselves in the foot 
                                                                               # if they enable page-locked memory for everything, because it cant be pre-empted. 
                                                                               # That is why we did not make it default True
                                                                               # If you are seeing system freeze or swap being used a lot, disable it.
                        num_workers=config['data_handler']['num_workers']           # Each worker loads a separate batch, not individual samples within a batch
            ),
            # 'val'   : torch.utils.data.DataLoader(
            #             image_datasets['val'], 
            #             batch_size=self.batch_size, 
            #             shuffle=False, 
            #             pin_memory=True,
            #             num_workers=config['data_handler']['num_workers']
            # ),
            'test'  : torch.utils.data.DataLoader(
                        image_datasets['test'], 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        pin_memory=True,
                        num_workers=config['data_handler']['num_workers']
            )
        }


        config['data_handler']['dataset_sizes'] = dataset_sizes = {x: len(image_datasets[x]) for x in self.dataset_split}
        config['data_handler']['class_names'] = class_names = image_datasets['train'].classes
        config['data_handler']['num_classes'] = num_classes = len(class_names)

        logging.debug('dataset_sizes {}'.format(dataset_sizes))
        logging.debug('class_names {}'.format(class_names))
        logging.debug('num_classes {}'.format(num_classes))


        ## Get class weights for calculating weighted loss

        # class_image_freqs=[]
        # for class_name in class_names:
        #     class_image_count = len(glob.glob(os.path.join(self.data_dir, self.data_name, 'train', class_name) + '/*.jpg'))
        #     class_image_freqs.append(class_image_count)
        # logging.debug('class_image_freqs {}'.format(class_image_freqs))

        # class_weights=[]
        # for class_image_freq in class_image_freqs:
        #     class_weights.append(np.sum(class_image_freqs)/class_image_freq)
        # logging.debug('class_weights {}'.format(class_weights))

        # class_weights =  class_weights/np.sum(class_weights) 
        # logging.debug('class_weights {}'.format(class_weights))

        # class_weights = torch.from_numpy(class_weights)
        # class_weights = class_weights.float()
        # class_weights = class_weights.to(config['device'])

        # config['class_weights'] = class_weights

        # logging.debug('class_weights {}'.format(config['class_weights']))

