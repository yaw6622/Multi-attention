import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from osgeo import gdal

class JLDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        """
        初始化 JLDataset
        :param data_dir: 数据所在文件夹路径
        :param label_dir: 标签所在文件夹路径
        :param transform: 可选的数据增强或变换操作
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        self.transform = transform

        # 确保数据文件和标签文件数量一致
        assert len(self.data_files) == len(self.label_files), "数据和标签文件数量不匹配"

    def __len__(self):
        # 返回数据集的大小
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        根据索引提取数据和标签
        :param idx: 索引
        :return: 返回数据和标签
        """
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # 加载数据和标签
        data = np.load(data_path)
        label = np.load(label_path)

        # 转换为 PyTorch 张量
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # 应用数据增强（如果有）
        if self.transform:
            data = self.transform(data)

        return data, label


class OEMDataset(Dataset):
    def __init__(self, data_folder, label_folder, if_transform=False):
        """
        初始化数据集。
        参数:
        - data_folder: 存储图像数据的小块文件夹路径
        - label_folder: 存储标签数据的小块文件夹路径
        """
        self.data_folder = data_folder
        self.label_folder = label_folder

        # 获取数据文件名列表
        self.data_files = sorted(os.listdir(data_folder))
        self.label_files = sorted(os.listdir(label_folder))

        # 确保图像和标签数量一致
        assert len(self.data_files) == len(self.label_files), "数据和标签数量不一致"
        self.if_transform = if_transform
        self.mean = None
        self.std = None
        if self.if_transform:
            self.mean, self.std = self.calculate_mean_std()

        # 定义图像的转换
        if if_transform:
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为张量并归一化到[0, 1]
                transforms.Normalize(mean=self.mean,
                                     std=self.std) if self.mean is not None and self.std is not None else transforms.Lambda(
                    lambda x: x)  # 正则化
            ])

    def calculate_mean_std(self):
        """
        计算数据集中图像的均值和标准差。
        返回:
        - mean: 每个通道的均值
        - std: 每个通道的标准差
        """
        mean = 0.0
        std = 0.0
        total_images = len(self.data_files)

        for img_name in tqdm(self.data_files):
            img_path = os.path.join(self.data_folder, img_name)
            image = self.load_image(img_path)
            mean += image.mean(dim=[1, 2])  # 对每个通道求均值
            std += image.std(dim=[1, 2])  # 对每个通道求标准差

        mean /= total_images
        std /= total_images
        return mean.tolist(), std.tolist()  # 返回列表形式

    def load_image(self, img_path):
        """
        使用gdal加载图像。
        参数:
        - img_path: 图像文件路径
        返回:
        - image: 转换为torch.Tensor格式的图像
        """
        dataset = gdal.Open(img_path)
        image = dataset.ReadAsArray()  # 读取为numpy数组，形状为 [C, H, W]

        # 转换为torch.Tensor并确保形状为 [3, H, W]
        image = torch.tensor(image, dtype=torch.float32)
        if image.shape[0] == 3:  # 确保图像是三通道
            return image
        else:
            raise ValueError(f"图像 {img_path} 的通道数不为3。")

    def load_label(self, lbl_path):
        """
        使用gdal加载标签。
        参数:
        - lbl_path: 标签文件路径
        返回:
        - label: 转换为torch.Tensor格式并去掉单通道的标签
        """
        dataset = gdal.Open(lbl_path)
        label = dataset.ReadAsArray()  # 读取为numpy数组，形状为 [1, H, W] 或 [H, W]

        # 转换为torch.Tensor并确保形状为 [H, W]
        label = torch.tensor(label, dtype=torch.long)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)  # 去除单通道维度
        return label

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        根据索引返回图像和标签。
        参数:
        - idx: 索引
        返回:
        - 图像和标签张量
        """
        # 获取图像和标签的文件路径
        img_path = os.path.join(self.data_folder, self.data_files[idx])
        lbl_path = os.path.join(self.label_folder, self.label_files[idx])

        # 加载图像和标签
        image = self.load_image(img_path)
        label = self.load_label(lbl_path)

        # 应用转换
        if self.if_transform:
            image = self.data_transform(image)
        else:
            image = image / 255.0  # 将图像归一化到[0, 1]

        return image, label

class PotsdamDataset(Dataset):
    def __init__(self, data_folder, label_folder, if_transform=False):
        """
        初始化数据集。
        参数:
        - data_folder: 存储图像数据的文件夹路径（tif格式）
        - label_folder: 存储标签数据的文件夹路径（npy格式）
        - if_transform: 是否进行数据增强和归一化（默认为 False）
        """
        self.data_folder = data_folder
        self.label_folder = label_folder

        # 获取数据文件名列表
        self.data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.tif')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.npy')])

        # 确保图像和标签数量一致
        assert len(self.data_files) == len(self.label_files), "数据和标签数量不一致"
        self.if_transform = if_transform
        self.mean = None
        self.std = None
        if self.if_transform:
            self.mean, self.std = self.calculate_mean_std()

        # 定义图像的转换
        if if_transform:
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为张量并归一化到[0, 1]
                transforms.Normalize(mean=self.mean,
                                     std=self.std) if self.mean is not None and self.std is not None else transforms.Lambda(
                    lambda x: x)  # 正则化
            ])

    def calculate_mean_std(self):
        """
        计算数据集中图像的均值和标准差。
        返回:
        - mean: 每个通道的均值
        - std: 每个通道的标准差
        """
        mean = 0.0
        std = 0.0
        total_images = len(self.data_files)

        for img_name in tqdm(self.data_files):
            img_path = os.path.join(self.data_folder, img_name)
            image = self.load_image(img_path)
            mean += image.mean(dim=[1, 2])  # 对每个通道求均值
            std += image.std(dim=[1, 2])  # 对每个通道求标准差

        mean /= total_images
        std /= total_images
        return mean.tolist(), std.tolist()  # 返回列表形式

    def load_image(self, img_path):
        """
        使用gdal加载图像。
        参数:
        - img_path: 图像文件路径
        返回:
        - image: 转换为torch.Tensor格式的图像
        """
        dataset = gdal.Open(img_path)
        image = dataset.ReadAsArray()  # 读取为numpy数组，形状为 [C, H, W]

        # 转换为torch.Tensor并确保形状为 [4, H, W]
        image = torch.tensor(image, dtype=torch.float32)
        if image.shape[0] == 4:  # 确保图像是4通道
            return image
        else:
            raise ValueError(f"图像 {img_path} 的通道数不为4。")

    def load_label(self, lbl_path):
        """
        使用npy加载标签。
        参数:
        - lbl_path: 标签文件路径
        返回:
        - label: 转换为torch.Tensor格式的标签
        """
        label = np.load(lbl_path)  # 读取为numpy数组，形状为 [H, W]
        label = torch.tensor(label, dtype=torch.long)  # 转换为torch.Tensor并确保是整型
        return label

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        根据索引返回图像和标签。
        参数:
        - idx: 索引
        返回:
        - 图像和标签张量
        """
        # 获取图像和标签的文件路径
        img_path = os.path.join(self.data_folder, self.data_files[idx])
        lbl_path = os.path.join(self.label_folder, self.label_files[idx])

        # 加载图像和标签
        image = self.load_image(img_path)
        label = self.load_label(lbl_path)

        # 应用转换
        if self.if_transform:
            image = self.data_transform(image)
        else:
            image = image / 255.0  # 将图像归一化到[0, 1]

        return image, label

class GIDDataset(Dataset):
    def __init__(self, data_folder, label_folder, if_transform=False):
        """
        初始化数据集。
        参数:
        - data_folder: 存储图像数据的文件夹路径（PNG格式）
        - label_folder: 存储标签数据的文件夹路径（PNG格式）
        - if_transform: 是否进行数据增强和归一化（默认为 False）
        """
        self.data_folder = data_folder
        self.label_folder = label_folder

        # 获取数据文件名列表
        self.data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.npy')])

        # 确保图像和标签数量一致
        assert len(self.data_files) == len(self.label_files), "数据和标签数量不一致"
        self.if_transform = if_transform
        if if_transform:
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为张量并归一化到[0, 1]
                # transforms.Normalize(mean=self.mean,
                #                      std=self.std) if self.mean is not None and self.std is not None else transforms.Lambda(
                #     lambda x: x)  # 正则化
            ])


        # self.label_transform = transforms.ToTensor()  # 标签转换为张量

    def load_image(self, img_path):
        """
        使用 PIL 加载图像。
        参数:
        - img_path: 图像文件路径
        返回:
        - 图像: (4, 224, 224) 的张量
        """

        image = Image.open(img_path).convert("RGBA")  # 确保图像为4通道
        image = np.array(image)  # 转换为 NumPy 数组
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if image.shape[0] == 4:  # 确保图像是4通道
            return image
        else:
            raise ValueError(f"图像 {img_path} 的通道数不为4。")  # 转换为 [C, H, W]

    def load_label(self, lbl_path):
        """
        使用 PIL 加载标签。
        参数:
        - lbl_path: 标签文件路径
        返回:
        - 标签: (224, 224) 的张量
        """

        label = np.load(lbl_path)  # 读取为numpy数组，形状为 [H, W]
        label = torch.tensor(label, dtype=torch.long)  # 转换为torch.Tensor并确保是整型
        return label

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        根据索引返回图像和标签。
        参数:
        - idx: 索引
        返回:
        - 图像: (4, 224, 224) 张量
        - 标签: (224, 224) 张量
        """
        # 获取图像和标签的文件路径
        img_path = os.path.join(self.data_folder, self.data_files[idx])
        lbl_path = os.path.join(self.label_folder, self.label_files[idx])

        # 加载图像和标签
        image = self.load_image(img_path)
        label = self.load_label(lbl_path)

        # 应用转换
        if self.if_transform:
            image = self.data_transform(image)
        else:
            image = image / 255.0

        return image, label

class PS_SS_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, time_stamp,transform=None, norm=True):
        """
        初始化数据集
        参数:
        - data_dir: 数据文件目录 (T, C, H, W 格式的 numpy 文件)
        - label_dir: 标签文件目录 (H, W 格式的 numpy 文件)
        - transform: 数据增强变换函数
        - norm: 是否对通道归一化
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.norm = norm
        self.time_stamp = time_stamp

        # 获取所有数据和标签文件的路径
        self.data_files = sorted(os.listdir(self.data_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        # 如果需要归一化，计算第12个时间点的均值和标准差
        if self.norm:
            self.mean, self.std = self._compute_channel_stats()

    def __getitem__(self, idx):
        """
        获取数据和标签
        参数:
        - idx: 数据索引
        返回:
        - data: 形状为 (C, H, W) 的第12个时间点的归一化数据
        - label: 对应的标签，形状为 (H, W)
        """
        data_file = os.path.join(self.data_dir, self.data_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])

        # 加载数据和标签
        data = np.load(data_file).astype(np.float32)  # 形状为 (T, C, H, W)
        label = np.load(label_file).astype(np.uint8)  # 形状为 (H, W)

        # 提取第T_S个时间点 (T维度的第T_S索引)
        data = data[self.time_stamp]  # 形状变为 (C, H, W)

        # 对每个通道进行归一化
        if self.norm:
            data = self._standardize_channels(data)

        # 数据增强变换
        if self.transform is not None:
            data, label = self.transform(data, label)

        return data, label

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_files)

    def _compute_channel_stats(self):
        """
        计算每个通道的均值和标准差，仅针对第12个时间点的数据。
        返回:
        - mean: 每个通道的均值 (列表)
        - std: 每个通道的标准差 (列表)
        """
        channel_stats = []
        for channel in range(10):  # 假设数据有10个通道
            channel_data = []
            for data_file in self.data_files:
                data = np.load(os.path.join(self.data_dir, data_file)).astype(np.float64)  # (T, C, H, W)
                channel_data.append(data[self.time_stamp, channel].flatten())  # 提取第12个时间点的特定通道数据
            channel_data = np.concatenate(channel_data, axis=0)
            channel_mean = np.mean(channel_data)
            channel_std = np.std(channel_data)
            channel_stats.append((channel_mean, channel_std))

        mean = [stat[0] for stat in channel_stats]
        std = [stat[1] for stat in channel_stats]

        return mean, std

    def _standardize_channels(self, data):
        """
        对每个通道进行标准化
        参数:
        - data: 单个时间点的通道数据，形状为 (C, H, W)
        返回:
        - standardized_data: 标准化后的数据，形状为 (C, H, W)
        """
        standardized_data = np.zeros_like(data, dtype=np.float32)
        for channel in range(data.shape[0]):  # 遍历每个通道
            standardized_data[channel] = (data[channel] - self.mean[channel]) / self.std[channel]
        return standardized_data

