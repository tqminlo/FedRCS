import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.vision import VisionDataset


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.targets = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.targets.append(label)

        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ICHDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['0', '1', '2', '3', "4"]
        self.image_paths = []
        self.targets = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.targets.append(label)

        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Cifar10Valid(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', "deer", "dog", "frog", "horse", "ship", "truck"]
        self.image_paths = []
        self.targets = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.targets.append(label)

        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Cifar10Pseudo(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.npy_paths = []

        for npy_name in os.listdir(root_dir):
            npy_path = os.path.join(root_dir, npy_name)
            self.npy_paths.append(npy_path)

        self.targets = torch.tensor(np.zeros(shape=(len(self.npy_paths,))))

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        # image = np.load(npy_path).astype(np.float32)
        image = Image.open(npy_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


if __name__ == "__main__":
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for i in range(10000):
        img_pseudo = np.random.randint(0, 255, size=(32,32,3))
        path = os.path.join("Cifar_pseudo2", f"{i:05}.jpg")
        cv2.imwrite(path, img_pseudo)
        # path = os.path.join("Cifar_pseudo", f"{i:05}.npy")
        # img = np.zeros((32, 32, 3), dtype=np.float32)
        # for c in range(3):
        #     img[..., c] = np.random.normal(loc=mean[c], scale=std[c], size=(32, 32))
        #     img = np.clip(img, 0, 1)
        #     img = np.round(img*255).astype(int)
        #     # print(np.min(img), np.max(img))
        # np.save(path, img)
        print("---")