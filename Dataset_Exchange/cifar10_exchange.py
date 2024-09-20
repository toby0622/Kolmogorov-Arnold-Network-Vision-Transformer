import numpy as np
import pickle
import imageio
import os


# 解壓縮，返回解壓後的字典
def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding="latin1")
    fo.close()
    return dict


list = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

for z in range(10):
    if not os.path.exists("CIFAR10/" + list[z]):
        os.makedirs("CIFAR10/" + list[z])

# 生成訓練集圖片，如果需要其他格式如 PNG，請更改圖片後綴名
for j in range(1, 6):
    dataName = (
        "C:/Users/toby0/Documents/GitHub/Kolmogorov-Arnold-Network-Vision-Transformer/Dataset_Exchange/CIFAR10_Batch/data_batch_"
        + str(j)
    )

    # 讀取當前目錄下的 data_batch_? 文件
    # dataName 即 data_batch_? 文件的路徑，此為本機上的絕對路徑
    Xtr = unpickle(dataName)

    print(dataName + " is loading...")

    for i in range(0, 10000):
        # Xtr['data'] 為圖片二進制數據
        img = np.reshape(Xtr["data"][i], (3, 32, 32))
        # 讀取 image
        img = img.transpose(1, 2, 0)
        picName = (
            "CIFAR10/"
            + list[Xtr["labels"][i]]
            + "/"
            + str(i + (j - 1) * 10000)
            + ".jpg"
        )  # png

        # Xtr['labels'] 為圖片的標籤，值範圍 0-9
        imageio.imwrite(picName, img)

    print(dataName + " loaded.")

print("All data batches loaded.")
