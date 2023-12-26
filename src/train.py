# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "训练"
__Created__ = 2023/12/14 16:25
"""
import pandas as pd
import torch.cuda
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import Dataset
from model import BertClassifier


def train(model, model_save_path, train_dataset, val_dataset, batch_size, lr, epochs):
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # 是否使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)

    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)

    best_avg_acc_val = 0
    for epoch in range(epochs):
        # 训练集损失&准确率
        total_loss_train = 0
        total_acc_train = 0
        model.train()
        # 训练进度
        for train_input, train_label in tqdm(train_loader):
            train_label = train_label.to(device)
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            # 模型输出
            output = model(input_ids, attention_mask)
            # 计算损失
            loss = criterion(output, train_label)
            total_loss_train += loss
            # 计算准确率
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            loss.backward()
            optim.step()

        # 模型验证
        total_loss_val = 0
        total_acc_val = 0
        # 验证无需梯度计算
        model.eval()
        with torch.no_grad():
            # 使用当前epoch训练好的模型验证
            for val_input, val_label in val_loader:
                val_label = val_label.to(device)
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)
                # 模型输出
                output = model(input_ids, attention_mask)
                loss = criterion(output, val_label)
                total_loss_val += loss
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        # save model
        if (total_acc_val / len(val_dataset)) > best_avg_acc_val:
            best_avg_acc_val = total_acc_val / len(val_dataset)
            torch.save(model.state_dict(), model_save_path)
            print(f'''best model | Val Accuracy: {best_avg_acc_val: .3f}''')
        print(
            f'''Epochs: {epoch + 1} 
              | Train Loss: {total_loss_train / len(train_dataset): .3f} 
              | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
              | Val Loss: {total_loss_val / len(val_dataset): .3f} 
              | Val Accuracy: {total_acc_val / len(val_dataset): .3f}''')


def test(model, model_save_path, test_dataset, batch_size):
    # 加载最佳模型权重
    model.load_state_dict(torch.load(model_save_path))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model = model.to(device)

    total_acc_test = 0
    model.eval()
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            attention_mask = test_input['attention_mask'].to(device)
            # model要求输入的矩阵(hidden_size,sequence_size),需要把第二纬度去除.squeeze(1)
            input_ids = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_ids, attention_mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_dataset): .3f}')


if __name__ == '__main__':
    batch_size = 24
    learn_rate = 1e-5
    epochs = 5
    # 加载数据
    label_datas = pd.read_excel('../train-datas/ChnSentiCorp_htl_all.xlsx')
    # 初始化dataset
    dateset = Dataset(label_datas, '../bert-base-chinese')
    # 创建模型
    model = BertClassifier('../bert-base-chinese')
    # 分割数据集
    total_size = len(label_datas)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(dateset, [train_size, val_size, test_size])
    print('train begin')
    train(model, '../result-model/classifier-model.pkl', train_dataset, val_dataset, batch_size, learn_rate, 5)
    print('train finish')
    print('test begin')
    test(model, '../result-model/classifier-model.pkl', test_dataset, batch_size)
    print('test finish')
