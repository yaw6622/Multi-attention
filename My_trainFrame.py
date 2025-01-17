# train.py
import json
import time
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from My_function.Make_Dataset import *
import load_model
from torchvision import transforms
from torch import nn
import pickle as pkl
from My_function.Func_use import *
from My_function.weight_init import weight_init
import pprint
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="ViM",
    type=str,
    help="Which model to use. Can be one of: (APTATT/ViTSeg/ViM/Unet/FPN/Linknet/DeeplabV3+/Segformer/AptSegV2)",
)
parser.add_argument(
    "--PR_Backbone",
    default="resnet101",
    type=str,
    help="Which pre_trained backbone to use. Can be one of: (resnet101/vgg16/mit_b2)",
)
parser.add_argument(
    "--dataset",
    default="JM",
    type=str,
    help="Which dataset to use. Can be one of: (JL_CF/OEM/Potsdam/Re_GID/PASTIS_SS/JM).",
)
parser.add_argument(
    "--res_dir",
    default="result_dir",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of workers"
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)

# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
parser.add_argument("--lr", default=0.002, type=float, help="Learning rate")
parser.add_argument("--ignore_index", default=None)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

parser.add_argument(
    "--num_classes",
    default=9,
    type=int,
    help="define number of classes",
)
parser.add_argument(
    "--channels",
    default=3,
    type=int,
    help="define input channels",
)
parser.add_argument(
    "--emd_dim",
    default=96,
    type=int,
    help="define embedded dimensions",
)
parser.add_argument(
    "--patch_size",
    default=4,
    type=int,
    help="define patch size",
)
parser.add_argument(
    "--img_size",
    default=224,
    type=int,
    help="define img size",
)
def iterate(
    model, data_loader, criterion, config, optimizer=None, mode="train", device=None,
):
    loss_meter = AverageMeter()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        device=config.device,
    )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        x,  y = batch
        x = x.float()
        y = y.long()

        if mode != "train":
            if config.model == "APTATT":
                with torch.no_grad():
                    out, weight_alpha, weight_beta = model(x)
                    weight_alpha = weight_alpha.detach().cpu().item()
                    weight_beta = weight_beta.detach().cpu().item()


            else:
                with torch.no_grad():
                    out= model(x)
        else:
            if config.model == "APTATT":
                optimizer.zero_grad()
                out, weight_alpha, weight_beta = model(x)
                weight_alpha = weight_alpha.detach().cpu().item()
                weight_beta = weight_beta.detach().cpu().item()

            else:
                optimizer.zero_grad()
                out= model(x)

        loss = criterion(out, y)

        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, y)
        loss_meter.update(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            # print(
            #     "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
            #         i + 1, len(data_loader), loss_meter.value(), acc, miou
            #     )
            # )
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}%, mIoU {:.2f}%".format(
                    i + 1, len(data_loader),
                    loss_meter.value(),
                    acc * 100,  # 转换为百分比形式
                    miou * 100  # 转换为百分比形式
                )
            )

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()

    if config.model == "APTATT":
        metrics = {
            "{}_accuracy".format(mode): acc * 100,
            "{}_loss".format(mode): loss_meter.value(),
            "{}_IoU".format(mode): miou * 100,
            "{}_epoch_time".format(mode): total_time,
            "{}_weight_alpha".format(mode): weight_alpha,
            "{}_weight_beta".format(mode): weight_beta,
        }

    else:
        metrics = {
            "{}_accuracy".format(mode): acc * 100,
            "{}_loss".format(mode): loss_meter.value(),
            "{}_IoU".format(mode): miou * 100,
            "{}_epoch_time".format(mode): total_time,
        }

    if mode == "test":
        return metrics, iou_meter.confusion_matrix()  # Return confusion matrix in test mode
    else:
        return metrics

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


# Functions used to evaluate or record the model
def checkpoint(log, config):
    """保存训练日志到文件中"""
    log_file_path = os.path.join(config.res_dir, "trainlog.json")
    try:
        with open(log_file_path, "w") as outfile:
            json.dump(log, outfile, indent=4)
        print(f"Checkpoint saved: {log_file_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def print_metrics(metrics, mode="val"):
    """打印训练、验证或测试的指标"""
    print(f"{mode.capitalize()} Loss: {metrics[f'{mode}_loss']:.4f}, "
          f"Acc: {metrics[f'{mode}_accuracy']:.2f}, "
          f"IoU: {metrics[f'{mode}_IoU']:.4f}")

def save_best_model(epoch, model, optimizer, config):
    """保存最优模型"""
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, os.path.join(config.res_dir, "model.pth.tar"))
    print(f"Best model saved at epoch {epoch}.")

def save_results(metrics, conf_mat, config):
    with open(
        os.path.join(config.res_dir, "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "conf_mat.pkl"), "wb"
        ),
    )

def overall_performance(config):
    # 从文件读取混淆矩阵
    conf_mat_path = os.path.join(config.res_dir, "conf_mat.pkl")
    with open(conf_mat_path, 'rb') as f:
        conf_mat = pkl.load(f)

    # 去除 ignore_index 类别的影响
    if config.ignore_index is not None:
        conf_mat = np.delete(conf_mat, config.ignore_index, axis=0)
        conf_mat = np.delete(conf_mat, config.ignore_index, axis=1)

    # TP: True Positives, FP: False Positives, FN: False Negatives
    TP = np.diag(conf_mat)
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP
    TN = conf_mat.sum() - (FP + FN + TP)

    # 计算每个类别的 IoU
    IoU_per_class = TP / (TP + FP + FN + 1e-10)

    # Overall Accuracy (OA)
    OA = np.sum(TP) / np.sum(conf_mat)

    # Mean Intersection over Union (mIoU)
    mIoU = np.mean(IoU_per_class)

    # 每个类别的 Recall 和 Precision
    Recall_per_class = TP / (TP + FN + 1e-10)
    Precision_per_class = TP / (TP + FP + 1e-10)

    # Macro Recall and Macro Precision
    Macro_Recall = np.mean(Recall_per_class)
    Macro_Precision = np.mean(Precision_per_class)

    # F1 Score per class and Macro F1 Score
    F1_per_class = 2 * (Precision_per_class * Recall_per_class) / (Precision_per_class + Recall_per_class + 1e-10)
    Macro_F1 = np.mean(F1_per_class)

    # 打印并记录结果
    performance_metrics = {
        "Overall Accuracy (%)": OA * 100,
        "Mean IoU (%)": mIoU * 100,
        "Class-wise IoU (%)": (IoU_per_class * 100).tolist(),
        "Macro Recall (%)": Macro_Recall * 100,
        "Macro Precision (%)": Macro_Precision * 100,
        "Macro F1 (%)": Macro_F1 * 100,
    }

    print("Overall Accuracy: {:.2f}%".format(OA * 100))
    print("Mean IoU: {:.2f}%".format(mIoU * 100))
    print("Per-class IoU:", (IoU_per_class * 100).tolist())
    print("Macro Recall: {:.2f}%".format(Macro_Recall * 100))
    print("Macro Precision: {:.2f}%".format(Macro_Precision * 100))
    print("Macro F1: {:.2f}%".format(Macro_F1 * 100))

    # 保存结果到JSON文件
    performance_file = os.path.join(config.res_dir, "Overall_performance.json")
    with open(performance_file, 'w') as f:
        json.dump(performance_metrics, f, indent=4)

    print("Performance metrics saved to:", performance_file)

def main(config):
    device = torch.device(config.device)
    if config.dataset == "JL_CF":
        config.img_size = 224
        config.channels = 3
        config.num_classes = 8
        dataset_folder = 'JL_1_Dataset'
        img_train = os.path.join(dataset_folder, 'seg224_train')
        label_train = os.path.join(dataset_folder, 'label_train')
        img_test = os.path.join(dataset_folder, 'seg224_test')
        label_test = os.path.join(dataset_folder, 'label_test')
        img_val = os.path.join(dataset_folder, 'seg224_val')
        label_val = os.path.join(dataset_folder, 'label_val')
        transform = None
        dt_train = JLDataset(img_train, label_train, transform=transform)
        trainloader = DataLoader(dt_train, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_val = JLDataset(img_val, label_val, transform=transform)
        valloader = DataLoader(dt_val, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_test = JLDataset(img_test, label_test, transform=transform)
        testloader = DataLoader(dt_test, batch_size=config.batch_size, shuffle=True,drop_last=True)

    elif config.dataset == "OEM":
        config.img_size = 256
        config.channels = 3
        config.num_classes = 9
        dataset_folder = 'OpenEarthMap/seg256'
        img_train = os.path.join(dataset_folder, 'train_data')
        label_train = os.path.join(dataset_folder, 'train_label')
        img_test = os.path.join(dataset_folder, 'test_data')
        label_test = os.path.join(dataset_folder, 'test_label')
        img_val = os.path.join(dataset_folder, 'val_data')
        label_val = os.path.join(dataset_folder, 'val_label')
        if_transform = False
        dt_train = OEMDataset(img_train, label_train, if_transform=if_transform)
        trainloader = DataLoader(dt_train, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_val = OEMDataset(img_val, label_val, if_transform=if_transform)
        valloader = DataLoader(dt_val, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_test = OEMDataset(img_test, label_test, if_transform=if_transform)
        testloader = DataLoader(dt_test, batch_size=config.batch_size, shuffle=True,drop_last=True)
    elif config.dataset == "Potsdam":
        config.img_size = 256
        config.channels = 4
        config.num_classes = 6
        img_train = 'Potsdam/PD_data/train'
        label_train = 'Potsdam/PD_label/train'
        img_test = 'Potsdam/PD_data/test'
        label_test = 'Potsdam/PD_label/test'
        img_val = 'Potsdam/PD_data/val'
        label_val = 'Potsdam/PD_label/val'
        if_transform = False
        dt_train = PotsdamDataset(img_train, label_train, if_transform=if_transform)
        trainloader = DataLoader(dt_train, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_val = PotsdamDataset(img_val, label_val, if_transform=if_transform)
        valloader = DataLoader(dt_val, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_test = PotsdamDataset(img_test, label_test, if_transform=if_transform)
        testloader = DataLoader(dt_test, batch_size=config.batch_size, shuffle=True,drop_last=True)

    elif config.dataset == "Re_GID":
        config.img_size = 224
        config.channels = 4
        config.num_classes = 20
        dataset_folder = 'Re_GID'
        img_train = os.path.join(dataset_folder, 'data_train')
        label_train = os.path.join(dataset_folder, 'label_train')
        img_test = os.path.join(dataset_folder, 'data_test')
        label_test = os.path.join(dataset_folder, 'label_test')
        img_val = os.path.join(dataset_folder, 'data_val')
        label_val = os.path.join(dataset_folder, 'label_val')
        if_transform = False
        dt_train = GIDDataset(img_train, label_train, if_transform=if_transform)
        trainloader = DataLoader(dt_train, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_val = GIDDataset(img_val, label_val, if_transform=if_transform)
        valloader = DataLoader(dt_val, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_test = GIDDataset(img_test, label_test, if_transform=if_transform)
        testloader = DataLoader(dt_test, batch_size=config.batch_size, shuffle=True,drop_last=True)
    elif config.dataset == "PASTIS_SS":
        config.img_size = 128
        config.channels = 10
        config.num_classes = 16
        img_train = 'PASTIS_SS/SSPAS_24/24Data_train'
        label_train = 'PASTIS_SS/SSPAS_24_LABEL/train_label'
        img_test = 'PASTIS_SS/SSPAS_24/24Data_test'
        label_test = 'PASTIS_SS/SSPAS_24_LABEL/test_label'
        img_val = 'PASTIS_SS/SSPAS_24/24Data_val'
        label_val = 'PASTIS_SS/SSPAS_24_LABEL/val_label'
        if_transform = False
        dt_train = PS_SS_Dataset(img_train, label_train, norm=if_transform)
        trainloader = DataLoader(dt_train, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_val = PS_SS_Dataset(img_val, label_val, norm=if_transform)
        valloader = DataLoader(dt_val, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_test = PS_SS_Dataset(img_test, label_test, norm=if_transform)
        testloader = DataLoader(dt_test, batch_size=config.batch_size, shuffle=True,drop_last=True)
    elif config.dataset == "JM":
        config.img_size = 128
        config.channels = 10
        config.num_classes = 3
        img_train = 'JM_Dataset/Jingmen_Data/Jingmen_Train'
        label_train = 'JM_Dataset/Jingmen_Label/Label_Train'
        img_test = 'JM_Dataset/Jingmen_Data/Jingmen_Test'
        label_test = 'JM_Dataset/Jingmen_Label/Label_Test'
        img_val = 'JM_Dataset/Jingmen_Data/Jingmen_Val'
        label_val = 'JM_Dataset/Jingmen_Label/Label_Val'
        if_transform = False
        time_stamp = 4
        dt_train = PS_SS_Dataset(img_train, label_train, time_stamp,norm=if_transform)
        trainloader = DataLoader(dt_train, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_val = PS_SS_Dataset(img_val, label_val, time_stamp, norm=if_transform)
        valloader = DataLoader(dt_val, batch_size=config.batch_size, shuffle=True,drop_last=True)
        dt_test = PS_SS_Dataset(img_test, label_test, time_stamp, norm=if_transform)
        testloader = DataLoader(dt_test, batch_size=config.batch_size, shuffle=True,drop_last=True)
    print(
        "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
    )
    # load model
    model = load_model.get_model(config, mode="semantic")
    config.N_Parameter = N_Parameter(model)
    if config.model == "ViM":
        print('for an unknown reason, Vim model cannot be printed out')
    else:
        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))

        print(model)

    print("TOTAL TRAINABLE PARAMETERS :", config.N_Parameter)
    print("Trainable layers:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    model = model.to(device)
    model.apply(weight_init)
    # load saved model if needed

    # sd = torch.load(
    #     os.path.join(config.res_dir, "model.pth.tar"),
    #     map_location=device,
    # )
    # model.load_state_dict(sd["state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    weights = torch.ones(config.num_classes, device=device).float()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # 训练日志和记录最优的 IoU
    trainlog = {}
    best_mIoU = 0

    # 训练和验证循环
    for epoch in range(1, config.epochs + 1):
        print(f"EPOCH {epoch}/{config.epochs}")

        # 训练模式
        model.train()
        train_metrics = iterate(
            model=model,
            data_loader=trainloader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="train",
            device=device,
        )

        # 每隔 config.val_every 个 epoch 进行验证
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("Validation . . .")
            model.eval()
            with torch.no_grad():
                val_metrics = iterate(
                    model=model,
                    data_loader=valloader,
                    criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    mode="val",
                    device=device,
                )

            # 打印验证结果
            print_metrics(val_metrics, mode="val")

            # 保存训练和验证的日志
            trainlog[epoch] = {**train_metrics, **val_metrics}

            # 保存模型检查点
            checkpoint(trainlog, config)

            # 保存最佳模型
            if val_metrics["val_IoU"] >= best_mIoU:
                best_mIoU = val_metrics["val_IoU"]
                save_best_model(epoch, model, optimizer, config)

        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(trainlog, config)

    print("Testing best epoch . . .")
    model.load_state_dict(torch.load(os.path.join(config.res_dir, "model.pth.tar"))["state_dict"])
    model.eval()
    with torch.no_grad():
        test_metrics, conf_mat = iterate(
            model=model,
            data_loader=testloader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="test",
            device=device,
        )

    # 打印测试结果
    print_metrics(test_metrics, mode="test")

    # 保存测试结果
    save_results(test_metrics, conf_mat.cpu().numpy(), config)

    # 评估总体性能
    overall_performance(config)



if __name__ == '__main__':
    config = parser.parse_args()
    pprint.pprint(config)
    main(config)
