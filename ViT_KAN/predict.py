import os
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # 加載圖片
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "文件: '{}' 不存在.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    # [N, C, H, W]
    img = data_transform(img)

    # 擴展批次維度
    img = torch.unsqueeze(img, dim=0)

    # 讀取 class_indict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "文件: '{}' 不存在.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 創建模型
    model = create_model(num_classes=5, has_logits=False).to(device)

    # 加載模型權重
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # 預測類別
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "類別: {}   機率: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy()
    )

    plt.title(print_res)

    for i in range(len(predict)):
        print(
            "類別: {:10}   機率: {:.3}".format(class_indict[str(i)], predict[i].numpy())
        )

    plt.show()


if __name__ == "__main__":
    main()
