import os
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from utils import *


if __name__ == '__main__':
    # data process
    data_dir = os.path.join('.', 'data', 'dog-breed-identification')
    data_dir = os.path.join('.', 'data', 'kaggle_dog_tiny')
    valid_ratio = 0.1
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

    # image augmentation
    transform_train = torchvision.transforms.Compose([
        # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
        # 然后，缩放图像以创建224 x 224的新图像
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        # 随机更改亮度，对比度和饱和度
        torchvision.transforms.ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4),
        # 添加随机噪声
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

    # fine-tunning net, parameter, device, optimizer, loss
    batch_size, num_epochs = 128, 10
    lr, wd = 1e-2, 1e-4
    lr_period, lr_decay = 2, 0.9
    param_group = True
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else [torch.device('cpu')]
    print('training on', devices[0])

    os.environ['TORCH_HOME'] = './models/pretrained_resnet'  # setting the environment variable
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False

    net = nn.DataParallel(finetune_net, device_ids=devices).to(devices[0])
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD((param for param in net.parameters()if param.requires_grad),
                          lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    # train and validation
    train_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'train'), transform=transform_train)
    valid_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'valid'), transform=transform_test)

    train_iter = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for X, y in tqdm(train_iter):
            X, y = X.to(devices[0]), y.to(devices[0])
            optimizer.zero_grad()
            ls = loss(net(X), y)
            ls.backward()
            optimizer.step()
            metric.add(ls * X.shape[0], y.shape[0])
        scheduler.step()
        train_ls = metric[0] / metric[1]
        valid_ls = evaluate_loss(net, loss, valid_iter, devices)
        print(f'train loss {train_ls:.3f}, valid loss {valid_ls:.3f}')

    # train the model
    train_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'train_valid'), transform=transform_train)
    test_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'test'), transform=transform_test)

    train_iter = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for X, y in tqdm(train_iter):
            X, y = X.to(devices[0]), y.to(devices[0])
            optimizer.zero_grad()
            ls = loss(net(X), y)
            ls.backward()
            optimizer.step()
            metric.add(ls * X.shape[0], y.shape[0])
        scheduler.step()
        train_ls = metric[0] / metric[1]
        print(f'train loss {train_ls:.3f}')
    torch.save(net.state_dict(), './models/resnet34.pth')

    # predict
    # predict
    model = net
    model.load_state_dict(torch.load('./models/resnet34.pth', map_location=devices[0]))

    preds = []
    for data, _ in test_iter:
        output = F.softmax(net(data.to(devices[0])), dim=0)
        preds.extend(output.cpu().detach().numpy())
    ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
    with open('submission.csv', 'w') as f:
        f.write('id,' + ','.join(train_ds.classes) + '\n')
        for i, output in zip(ids, preds):
            f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')

