# 文本分类-bert

使用PyTorch基于bert-base预训练模型实现文本分类

## 基本步骤

### 下载bert-base预训练模型

BERT 是 **B**idirectional Encoder **R**epresentations from **T**ransformers 的首字母缩写词。一个基于 Transformer 的机器学习模型。BERT 架构由多个堆叠在一起的 Transformer 编码器组成。每个 Transformer 编码器都封装了两个子层：**一个自注意力层和一个前馈层。**

有两种不同的 BERT 模型：

1. BERT **base** 模型，由 12 层 Transformer 编码器、12 个注意力头、768 个隐藏大小和 110M 参数组成。
2. BERT **large** 模型，由 24 层 Transformer 编码器、16 个注意力头、1024 个隐藏大小和 340M 个参数组成。

BERT 是一个强大的语言模型至少有两个原因：

1. 它使用从 BooksCorpus （有 8 亿字）和 Wikipedia（有 25 亿字）中提取的未标记数据进行预训练。
2. 顾名思义，它是通过利用编码器堆栈的双向特性进行预训练的。这意味着 BERT 不仅从左到右，而且从右到左从单词序列中学习信息。

google bert源码地址：https://github.com/google-research/bert

bert预训练模型下载地址：https://huggingface.co/bert-base-uncased

### 准备label好的训练数据

数据集采集酒店评论分类数据。

```text
数据字段：
Label：1表示正向评论，0表示负向评论
Review：评论内容
```

数据集地址：https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv

其它数据集：https://zhuanlan.zhihu.com/p/80029681 https://github.com/CLUEbenchmark/CLUEDatasetSearch

### 预处理数据

1. 输入数据编码
   bert模型要求输入的数据是一系列的tokens(words)，即需要对输入的句子进行编码(encode)，这个过程称为`tokenization`。bert模型文件提供了`vocab.txt`文件，该文件是bert模型生成的字典编码表。代码示例如戏：

   ```python
   from transformers import BertTokenizer, BertModel
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   text = "Replace me by any text you'd like."
   encoded_input = tokenizer(text, return_tensors='pt')
   ```

2. 输入数据插入[CLS]和[SEP]

   ```python
   encoded_input = tokenizer(text, return_tensors='pt')
   ```

   上述代码会自动添加[CLS]和[SEP]

### 创建模型

创建作用于文本分类的下游模型（基于bert模型）

代码示例如下：

```python
class BertClassifier(nn.Module):
    def __init__(self, model_path, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

```

这个 `BertClassifier` 类是一个简单的基于 BERT 模型的分类器。以下是对其主要组件和功能的分析：

1. **构造函数 (`__init__`):**

   - `model_path`: 传入的 BERT 模型的路径或名称。
   - `dropout`: Dropout 层的概率，默认为 0.5。
   - `bert`: BERT 模型从预训练模型加载。
   - `dropout`: 一个 Dropout 层，用于防止过拟合。
   - `linear`: 一个线性层，将 BERT 的输出维度 (768) 转换为最终输出类别的数量 (2)。
   - `relu`: 一个 ReLU 激活函数，用于引入非线性。

2. **前向传播 (`forward`):**

   - `input_ids` 和 `attention_mask` 作为输入传递给 BERT 模型。
   - 从 BERT 输出中提取池化后的表示 (`pooled_output`)。
   - 通过 Dropout 层进行正则化。
   - 通过线性层进行转换，将特征映射到最终的输出类别空间。
   - 最后，通过 ReLU 激活函数获得最终的输出。

3. **`dropout`层的意义**

   Dropout 是一种用于神经网络的正则化技术，其目的是减少模型对训练数据的过拟合。过拟合是指模型在训练数据上表现得很好，但在未见过的数据上表现较差。Dropout 的引入旨在减少神经网络中节点之间的协作，强制网络学习更鲁棒的特征，从而提高泛化能力。

   Dropout 的工作原理如下：

   1. **随机失活节点：** 在每次训练迭代中，随机选择网络中的一些节点，并将它们的输出置零。这样，被选择的节点在该迭代中对于前向传播和反向传播的梯度更新都不起作用。节点的选择是随机的，通常以一定的概率（dropout 率）进行。
   2. **减少过拟合：** 由于在每个迭代中都会随机失活节点，模型不能依赖于特定节点的存在，强制模型学习更加鲁棒的特征。这有助于减少模型对训练数据的过拟合。
   3. **集成学习的效果：** 可以将每个训练迭代视为通过随机选择不同的节点集合进行训练的一个“子模型”。在测试时，将所有节点保留，并通过平均或投票等方式综合这些子模型的预测，从而提高模型的泛化性能。

   总体而言，Dropout 是一种强大的正则化技术，它有助于提高神经网络的鲁棒性，减少过拟合，以及提高模型的泛化性能。

### 训练模型

训练基于bert模型的文本分类模型。代码示例如下：

```python
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

    print('train begin')
    model.train()
    for epoch in range(epochs):
        # 训练集损失&准确率
        total_loss_train = 0
        total_acc_train = 0
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
        if total_acc_val / len(val_dataset) > best_acc_val / len(val_dataset):
            best_acc_val = total_acc_val / len(val_dataset)
            torch.save(model.state_dict(), model_save_path)
            print(f'''best model | Val Accuracy: {best_acc_val: .3f}''')
        print(
            f'''Epochs: {epoch + 1} 
              | Train Loss: {total_loss_train / len(train_dataset): .3f} 
              | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
              | Val Loss: {total_loss_val / len(val_dataset): .3f} 
              | Val Accuracy: {total_acc_val / len(val_dataset): .3f}''')
```

1. **数据加载器（DataLoader）：**
   - 使用 `DataLoader` 加载训练和验证数据集。`train_loader` 用于训练，`val_loader` 用于验证。
   - `batch_size` 参数指定每个批次的样本数。
   - `shuffle=True` 表示在每个 epoch 开始时打乱数据，有助于模型更好地学习。
2. **设备设置：**
   - 检查是否有可用的 GPU，如果有，则将模型和损失函数移至 GPU 设备。
3. **损失函数和优化器：**
   - 定义交叉熵损失函数 `nn.CrossEntropyLoss()` 用于分类问题。
   - 定义 Adam 优化器，用于更新模型参数。
4. **训练循环（Training Loop）：**
   - 迭代训练数据集，每个 epoch 进行一次完整的训练循环。
   - 对于每个训练样本，进行前向传播、损失计算、反向传播和参数更新。
   - 通过 `torch.no_grad()` 关闭梯度计算，进行验证集的模型评估。
5. **模型保存：**
   - 如果验证集上的准确率优于之前的最佳准确率，则保存当前模型参数。这是一个简单的模型保存逻辑，实际应用中可能需要更复杂的逻辑，例如保存多个最佳模型或定期保存模型。
6. **训练和验证指标计算：**
   - 计算每个 epoch 的平均训练损失、训练准确率、验证损失和验证准确率。
   - 输出训练和验证指标，以监视模型的训练进度。

### 数据分类

训练时，需要对label好对数据集进行分割，一般包含(训练数据集[train-dataset]、验证数据集[val-dataset]、测试数据集[test-dataset])。代码示例如下：

```python
# 分割数据集
total_size = len(label_datas)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
# 分割数据集
train_dataset, val_dataset, test_dataset = random_split(dateset, [train_size, val_size, test_size])
```

### 模型保存

通常训练跑完一个epoch后，会对模型进行评估，此时使用的是验证数据集，同时关闭梯度计算。如果验证集上的准确率优于之前的最佳准确率，则保存当前模型参数。代码示例如下：

```python
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
        if total_acc_val / len(val_dataset) > best_acc_val / len(val_dataset):
            best_acc_val = total_acc_val / len(val_dataset)
            torch.save(model.state_dict(), model_save_path)
            print(f'''best model | Val Accuracy: {best_acc_val / len(val_dataset): .3f}''')
```

### 模型测试

训练完成保存最佳模型后，可对模型进行测试，此时使用的是测试数据集。代码示例如下：

```python
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
            input_ids = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_ids, attention_mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_dataset): .3f}')
```

