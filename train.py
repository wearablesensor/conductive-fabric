import random
import sys
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
import pandas as pd
import numpy as np
from pathlib import Path
import re
import glob
import shutil
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import os
from cnn_lstm_attention import cnn_Lstm_att

#更新模型训练路径
def increment_path(path, exist_ok=False, sep='', mkdir=True):
    """
    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    :param path: file or directory path to increment
    :param exist_ok: existing project/name ok, do not increment
    :param sep: separator for directory name
    :param mkdir: create directory
    :return: incremented path
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path
# Utility function to print the confusion matrix
def draw_confusion_matrix(label_true, label_pred, title,label_name, normlize, pdf_save_path, dpi,figsize=(4, 4),fontsize=12):
    cm = confusion_matrix(label_true, label_pred)
    if normlize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.imshow(cm, cmap='Greens', vmin=0, vmax=1)
    tick_marks = np.arange(len(label_name))
    plt.xticks(tick_marks, label_name, rotation=45,fontweight='bold',fontsize=fontsize)
    plt.yticks(tick_marks, label_name,fontweight='bold',fontsize=fontsize)
    plt.title(title, fontweight='bold')
    # plt.tight_layout()
    #
    # plt.colorbar(ticks=np.linspace(0, 1, 11),fontsize=fontsize)
    cbar = plt.colorbar(ticks=np.linspace(0, 1, 11))
    cbar.ax.tick_params(labelsize=fontsize, width=2, length=6, direction='out')
    cbar.ax.tick_params(which='major', pad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_weight('bold')


    fmt = '.2f' if normlize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",fontweight='bold',fontsize=fontsize)

    plt.ylabel('True label',fontweight='bold',fontsize=fontsize)
    plt.xlabel('Predicted label',fontweight='bold',fontsize=fontsize)

    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


# Loss Function
def criterion(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)



''' Training Process '''

# define function to create a single step of the training phase

def make_train_step(model, criterion, optimizer):
    # define the training step of the training phase
    def train_step(X, Y):
        # forward pass

        output_logits, output_softmax = model.forward(X)


        predictions = torch.argmax(output_softmax, dim=1)

        accuracy = torch.sum(Y == predictions) / float(len(Y))

        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y)

        # loss = loss.requires_grad_()
        # compute gradients for the optimizer to use
        loss.backward()

        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()
        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad()
        return loss.item(), accuracy * 100
    return train_step

def make_validate_fnc(model, criterion):
    def validate(X, Y):
        # don't want to update any network parameters on validation passes: don't need gradient
        # wrap in torch.no_grad to save memory and compute in validation phase:
        with torch.no_grad():
            # set model to validation phase i.e. turn off dropout and batchnorm layers
            model.eval()
            # get the model's predictions on the validation set
            output_logits, output_softmax = model.forward(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits, Y)

        return loss.item(), accuracy * 100, Y.cpu().numpy(),predictions.cpu().numpy(),predictions

    return validate

def make_save_checkpoint():
    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
    return save_checkpoint

def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

path = 'runs/train/exp'
def train(optimizer, model, num_epochs, X_train, Y_train, X_valid, Y_valid):
    # 保存best.pt的路径
    pth_path = increment_path(path)
    max_acc = 0
    for epoch in range(num_epochs):
        # set model to train phase
        model.train()
        # shuffle entire training set in each epoch to randomize minibatch order
        #随机生成顺序
        train_indices = np.random.permutation(train_size)

        test_indices = np.random.permutation(test_size)
        #打乱训练集顺序
        # shuffle the training set for each epoch:
        X_train = X_train[train_indices, :, :]

        # x_t = x_t[train_indices, :, :]
        Y_train = Y_train[train_indices]
        X_valid = X_valid[test_indices, :, :]

        # x_t = x_t[train_indices, :, :]
        Y_valid = Y_valid[test_indices]
        # print("shape is::::::::::")
        # print(X_train.shape)
        # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate
        epoch_acc = 0
        epoch_loss = 0
        #一个epoch的训练批次，训练集的大小/batchsize
        num_iterations = int(train_size / minibatch)
        start = time.time()
        # create a loop for each minibatch of 32 samples:
        for i in range(num_iterations):
            # track minibatch position based on iteration number:
            #一个训练批次的起始位置
            batch_start = i * minibatch
            # ensure we don't go out of the bounds of our training set:
            #一个训练批次的终止位置
            batch_end = min(batch_start + minibatch, train_size)
            # ensure we don't have an index error
            #一个训练批次的实际大小
            actual_batch_size = batch_end - batch_start

            X = X_train[batch_start:batch_end, :, :]

            Y = Y_train[batch_start:batch_end]

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=device).float()
            # x_tensor = torch.tensor(x, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)

            # Pass input tensors thru 1 training step (fwd+backwards pass)
            Y_tensor = torch.argmax(Y_tensor, dim=1)

            loss, acc = train_step(X_tensor, Y_tensor)

            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size

            # keep track of the iteration to see if the model's too slow
            print('\r' + f'Epoch {epoch}: iteration {i}/{num_iterations}', end='')
        end = time.time()
        time_interval = end-start
        print("Training use {:.0f}s".format(time_interval % 60))
        # create tensors from validation set

        X_valid_tensor = torch.tensor(X_valid, device=device).float()

        # x_tensor = torch.tensor(x_v, device=device).float()
        Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long, device=device)

        # calculate validation metrics to keep track of progress; don't need predictions now
        Y_valid_tensor = torch.argmax(Y_valid_tensor, dim=1)
        valid_loss, valid_acc, label_true ,label_pre,_ = validate(X_valid_tensor, Y_valid_tensor)

        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
        train_acces.append(epoch_acc.cpu())
        valid_acces.append(valid_acc.cpu())
        if  valid_acc.cpu() > max_acc:
             max_acc = valid_acc.cpu()
        # Save checkpoint of the model
             checkpoint_filename = os.path.join(pth_path,'{:s}.pt'.format('best'))
             confusion_filename = os.path.join(pth_path,'{:s}.jpg'.format('Confusion_Matrix'))
             result_filename = os.path.join(pth_path, '{:s}.txt'.format(model.__class__.__name__))
             # 如果文件已经存在，则删除文件
             if os.path.exists(checkpoint_filename):
                 os.remove(checkpoint_filename)
             if os.path.exists(confusion_filename):
                 os.remove(confusion_filename)
             save_checkpoint(optimizer, model, epoch, checkpoint_filename)
             # 清空画布
             plt.clf()
             draw_confusion_matrix(label_true=label_true,  # y_gt=[0,5,1,6,3,...]
                                   label_pred=label_pre,  # y_pred=[0,5,1,6,3,...]
                                   label_name=['stand', 'walk','run','sit','dribble','kick'], normlize=True,
                                   title="Our Method's Confusion Matrix",
                                   pdf_save_path=confusion_filename,
                                   dpi=500)

             # 重定向标准输出流到文件
             sys.stdout = open(result_filename, 'w')
             print('Macro precision:', precision_score(label_true, label_pre, average='macro'))
             print('Macro recall:', recall_score(label_true, label_pre, average='macro'))
             print('Macro f1-score:', f1_score(label_true, label_pre, average='macro'))
             print('Validation accuracy:',valid_acc.cpu())
             # 关闭文件
             sys.stdout.close()

             # 恢复标准输出流
             sys.stdout = sys.__stdout__

        # keep track of each epoch's progress

        print(f'\nEpoch {epoch} --- Train loss:{epoch_loss:.3f}, Train accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')
        print(""
              "")
    #将损失图和准确率图保存到文件
    pltshow(pth_path)

def pltshow(curve_path):
    # 设置全局字体属性和字体大小
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['font.size'] = 10

    plt.subplot(1, 2, 1)
    plt.title('Loss Curve', fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(train_losses[:], 'b')
    plt.plot(valid_losses[:], 'r')
    plt.legend(['Training loss', 'Test loss'], loc='upper right')
    curve_loss_path = os.path.join(curve_path, 'Lossfig.png')
    plt.savefig(curve_loss_path)

    plt.subplot(1, 2, 2)
    plt.title('Accuracy Curve', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(train_acces[:], 'b')
    plt.plot(valid_acces[:], 'r')
    plt.legend(['Training accuracy', 'Test accuracy'], loc='lower right')
    curve_acc_path = os.path.join(curve_path, 'Accuracy.png')
    plt.savefig(curve_acc_path)
    plt.tight_layout()
    plt.show()


def lodedata(folder_path):
    labels = []
    matrices = []

    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # 检查文件名是否包含特定关键词，确定标签
        if 'stand' in file_name:
            true_label = 0
        elif 'walk' in file_name:
            true_label = 1
        elif 'run' in file_name:
            true_label = 2
        elif 'sit' in file_name:
            true_label = 3
        elif 'dribble' in file_name:
            true_label = 4
        elif 'kict' in file_name:
            true_label = 5
        else:
            continue

        df = pd.read_csv(file_path, header=None)
        num_samples = int(len(df) // 50) # 计算样本数量
        print(num_samples)

        # 将每个样本存储到matrices中
        for i in range(num_samples):
            start_index = i *50
            end_index = (i + 1) * 50
            sample_data = df.iloc[start_index:end_index, :].values
            # print(sample_data.shape)
            scaler = preprocessing.StandardScaler()
            sample_data = scaler.fit_transform(sample_data)
            matrices.append(sample_data)
            # print(matrices)
            labels.append(true_label)

    X = np.stack(matrices, axis=0)
    Y = pd.get_dummies(labels).values

    return X, Y
if __name__ == '__main__':
    # 设置 Python 随机种子
    random.seed(5)

    # 设置 Numpy 随机种子
    np.random.seed(5)
    #
    # 设置 PyTorch 随机种子
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X,Y = lodedata(r'D:\Desktop\cwt')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # set device to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device ='cpu'
    print(torch.cuda.is_available())

    print(f'{device} selected')
    model = cnn_Lstm_att(4, 20, 1, 6, 3).to(device)
    print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-4)
    train_size = 1657
    test_size = 711
    # pick minibatch size (of 32... always)
    minibatch = 64
    epochs = 100
    # instantiate the checkpoint save function
    save_checkpoint = make_save_checkpoint()

    # instantiate the training step function
    train_step = make_train_step(model, criterion, optimizer=optimizer)

    # instantiate the validation loop function
    validate = make_validate_fnc(model, criterion)

    # instantiate lists to hold scalar performance metrics to plot later
    train_losses = []
    valid_losses = []
    train_acces = []
    valid_acces = []
    min_epoch = {}

    train(optimizer, model, epochs, X_train, Y_train, X_test, Y_test)