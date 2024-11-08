import torch
from PIL import Image
from torch.utils.data import Dataset


# 自定義資料集
class MyDataSet(Dataset):
    # 初始化資料集
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    # 返回資料集圖像的數量
    def __len__(self):
        return len(self.images_path)

    # 返回資料集中的圖像和標籤
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB 為彩色圖片，L 為灰度圖片
        if img.mode != "RGB":
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # 將資料集的圖像和標籤組合成一個 batch
    @staticmethod
    def collate_fn(batch):
        # PyTorch 官方實現的 default_collate 參考下列網址
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels
