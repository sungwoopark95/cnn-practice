import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from cfg import get_cfg

cfg = get_cfg()

if cfg.dataset == "cifar10":
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
elif cfg.dataset == "cifar100":
    labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
              'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 
              'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 
              'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 
              'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
              'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
              'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
              'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
              'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
              'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def plot_dataset(dataloader, grid_width=8, grid_height=2, figure_width=12, figure_height=3, y_hats=None):
    images, labels = next(iter(dataloader))
    f, ax = plt.subplots(grid_height, grid_width)
    f.set_size_inches(figure_width, figure_height)
    img_idx = 0
    for i in range(0, grid_height):
        for j in range(0, grid_width):
            image = images[img_idx]
            label = labels[img_idx]
            title_color = 'k'
            if y_hats is None:
                label_idx = int(label)
            else:
                label_idx = int(y_hats[img_idx])
                if int(labels[img_idx]) != label_idx:
                    title_color = 'r'
            label = labels[label_idx]
            ax[i][j].axis('off')
            ax[i][j].set_title(label, color=title_color)
            ax[i][j].imshow(np.transpose(image, (1, 2, 0)), aspect='auto')
            img_idx += 1
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)
    plt.show()
    
## augmentation
def gaussian_smoothing(image, filter_size=cfg.gs_f, sigma=cfg.gs_s):
    filter_size = int(filter_size)
    center = (filter_size-1)/2
    gaussian_filter = np.zeros((filter_size, filter_size))
    for row in range(filter_size):
        for col in range(filter_size):
            gaussian_filter[row, col] = np.exp((-(row-center) ** 2 - (col-center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    image = cv2.filter2D(image, -1, gaussian_filter)
    return image

def color_jitter(image):
    image = cv2.convertScaleAbs(image, alpha=cfg.cj_a, beta=cfg.cj_b)
    return image

def custom_augmentation(image):
    p = np.random.uniform(0, 1)

    if p < (1/3):
        ## move the image upward, downward, rightward, and leftward by one pixel
        direction = [(-1, 1), (1, -1), (-1, -1), (1, 1)]
        idx = np.random.choice(np.arange(4), size=1)[0]
        tx, ty = direction[idx]
        M = np.array([[1, 0, tx], 
                      [0, 1, ty]], dtype=np.float32)
        image = cv2.warpAffine(image, M, (0, 0))
    elif p < (2/3):
        ## horizontal flip
        image = cv2.flip(image, 1)
    else:
        ## rotate the image at most 15 degrees either clockwise or counter-clockwise
        rot_angle = int(cfg.ro_a)
        angle = np.random.choice(np.arange(-rot_angle, rot_angle))
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1])

    return image

## Dataset
class CustomDataset(Dataset):
    def __init__(self, train, prob=cfg.aug_p, data_dir="./data"):
        if cfg.dataset == "cifar10": 
            self.data = datasets.CIFAR10(root=data_dir, train=train, download=True)
        elif cfg.dataset == "cifar100":
            self.data = datasets.CIFAR100(root=data_dir, train=train, download=True)
        self.prob = prob

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        tf_methods = [gaussian_smoothing, color_jitter, custom_augmentation]
        
        p = np.random.uniform(0, 1)
        if p <= self.prob:
            method = np.random.choice(tf_methods, size=1)[0]
            image = method(image)
        else:
            image = image
            
        if cfg.img_size is not None:
            dim = (cfg.img_size, cfg.img_size)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def collate_fn(self, data):
        batch_x, batch_y = [], []

        for x, y in data:
            arr_x = np.array(x, dtype=np.uint8) # convert into numpy array
            timg_x = self.transform(arr_x)
            norm_x = timg_x / 255

            ## convert to tensor and transpose
            x_torch = torch.Tensor(np.array(norm_x, ndmin=4))
            x_torch = x_torch.permute(0,3,1,2)
            x_torch = np.squeeze(x_torch, axis=0)
            y_torch = torch.Tensor([y])

            batch_x.append(x_torch)
            batch_y.append(y_torch)
        
        batch_x = torch.stack(batch_x).float()
        batch_y = torch.cat(batch_y).long()

        return batch_x, batch_y