import os
import sys
import json
import pickle
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(112522083)  # 保證每次運行的隨機結果一致
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍歷所有文件夾，每個文件夾代表一個類別
    flower_class = [
        cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))
    ]

    # 排序，保證各平台順序一致
    flower_class.sort()

    # 生成類別名稱與對應的索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(
        dict((val, key) for key, val in class_indices.items()), indent=4
    )

    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    train_images_path = []  # 儲存訓練集的所有圖片路徑
    train_images_label = []  # 儲存訓練集圖片對應索引
    val_images_path = []  # 儲存驗證集的所有圖片路徑
    val_images_label = []  # 儲存驗證集圖片對應索引
    every_class_num = []  # 儲存每個類別的樣本數
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的圖片格式

    # 遍歷每個 Folder 的 File
    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        # 遍歷獲取 supported 支持的所有文件路徑
        images = [
            os.path.join(root, cla, i)
            for i in os.listdir(cla_path)
            if os.path.splitext(i)[-1] in supported
        ]

        # 排序，保證各平台順序一致
        images.sort()

        # 獲取該類別對應的索引
        image_class = class_indices[cla]

        # 紀錄該類別的樣本數
        every_class_num.append(len(images))

        # 依照比例隨機抽樣驗證集樣本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            # 如果該路徑在採樣的驗證集樣本中則存入驗證集
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            # 否則存入訓練集
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False

    if plot_image:
        # 繪製每個類別的樣本數柱狀圖
        plt.bar(range(len(flower_class)), every_class_num, align="center")
        # 將橫坐標 0, 1, 2, 3, 4 替換為相應的類別名稱
        plt.xticks(range(len(flower_class)), flower_class)

        # 在柱狀圖上添加數值標籤
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha="center")

        # 設置 x 坐標
        plt.xlabel("image class")
        # 設置 y 坐標
        plt.ylabel("number of images")
        # 設置柱狀圖的標題
        plt.title("flower class distribution")
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, "r")
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反 Normalize 操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去除 x 軸刻度
            plt.yticks([])  # 去除 y 軸刻度
            plt.imshow(img.astype("uint8"))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, "rb") as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)  # 累計預測正確的樣本數
    accu_loss = torch.zeros(1).to(device)  # 累計損失
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累計預測正確的樣本數
    accu_loss = torch.zeros(1).to(device)  # 累計損失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
