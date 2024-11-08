import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import torchprofile
import warnings
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate
from torchsummary import summary

# 忽略特定警告
warnings.filterwarnings("ignore", message="No handlers found: ")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = (
        read_split_data(args.data_path)
    )

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    }

    # 訓練資料集實例
    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )

    # 驗證資料集實例
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )

    batch_size = args.batch_size

    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # DataLoader使用的進程數

    print("Using {} Dataloader Workers Every Process".format(nw))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(
            args.weights
        )
        weights_dict = torch.load(args.weights, map_location=device)
        # 刪除不需要的權重
        del_keys = (
            ["head.weight", "head.bias"]
            if model.has_logits
            else [
                "pre_logits.fc.weight",
                "pre_logits.fc.bias",
                "head.weight",
                "head.bias",
            ]
        )
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除了 head 和 pre_logits 之外，其他權重全部凍結
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("Training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)

    # Scheduler 優化來自論文
    # Bag of Tricks for Image Classification with Convolutional Neural Networks (2018)
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )  # 餘弦
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()

        # 驗證
        val_loss, val_acc = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch
        )

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 訓練完成後評估模型參數和 FLOPs
    print("Model Parameters & FLOPs Evaluation")
    summary(model, (3, 224, 224))
    flops = torchprofile.profile_macs(model, torch.randn(1, 3, 224, 224).to(device))
    print(f"Total FLOPs: {flops / 1e9} GFLOPs")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)

    # 資料集根目錄
    parser.add_argument("--data-path", type=str, default="data/flower_photos")
    parser.add_argument("--model-name", default="", help="create model name")

    # 預訓練權重路徑，如果不想載入就設置為空字符串
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="initial weights path",
    )

    # 權重是否凍結
    parser.add_argument("--freeze-layers", type=bool, default=True)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)

    end_time = time.time()

    print("Training Time: {:.2f} (Minutes)".format((end_time - start_time) / 60))
