import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import argparse
from networks.ClassicNetwork.ResNet import ResNet50

import os
import numpy as np
import time

from dataloader import load_dataset

def save_file(path, list):
    f=open(path,'w')
    for line in list:
        f.write(line)
    f.close()

def main():  #每次需要修改model_name, class_number和dataset_dir
    model_name='OurModel'
    CLASS_NUMBER=2
    Best_ACCURACY=70
    dataset_dir = './dataset'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('device:', device)

    EPOCH = 200 #最大epoch数目
    pre_epoch = 0 #已训练epoch次数
    BATCH_SIZE = 10 #Batchsize
    LR = 2e-4 #学习率
    WEIGHT_DECAY = 5e-4 #衰减系数
    STEP_SIZE=50 #学习率衰减过程
    GAMMA=0.1 #The decay multiple in each decay step

    log_dir = './model/'+model_name+'/log'
    if os.path.exists(log_dir) is not True:
        os.makedirs(log_dir)
    source_dir = '../model/' + model_name + '/source'
    if os.path.exists(source_dir) is not True:
        os.makedirs(source_dir)
        #os.makedirs( './model/'+data_dir+'/source')
        #os.makedirs('./model/' + data_dir + '/result')
    # writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    net = ResNet50(num_classes=CLASS_NUMBER).to(device)
    #torch.load('./model/DR80/net.pkl')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    data_loader, data_size = load_dataset(BATCH_SIZE,data_dir=dataset_dir)
    if os.path.exists('./model/'+model_name+'/net.pkl') is not True:
        f_loss = open('./model/'+model_name+'/train_loss.txt', 'a')
        f_acc = open('./model/'+model_name+'/train_acc.txt', 'a')

        print("Start Training ResNet50")

        for epoch in range(pre_epoch, EPOCH):
            since = time.time()
            print('\n Epoch: {}'.format(epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            train_data = data_loader["train"]
            test_data = data_loader["test"]
            length = len(train_data)
            for i, data in enumerate(train_data, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward+backward
                outputs, features = net(inputs)
                #print(features)
                loss = criterion(outputs, labels)
                # print('features:', features.size())
                # print('outputs:',outputs.size())
                # print('labels:',labels)

                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                _, pre = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += torch.sum(pre == labels.data)
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% |time:%.3f'
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                         time.time() - since))
            scheduler.step(epoch)
            f_loss.write(str(float(sum_loss / (i + 1))) + '\n')
            f_acc.write(str(float(100. * correct / total)) + '\n')

            # testing

            if (epoch+1) % 1== 0:
                print('start to test')
                f_acc1 = open('./model/' + model_name + '/test_acc.txt', 'a')
                f_loss1 = open('./model/' + model_name + '/test_loss.txt', 'a')
                with torch.no_grad():  # 不用计算梯度，节省GPU
                    correct = 0
                    total = 0

                    labels_list = []
                    predited_list = []
                    preValue_list = []  # fpr
                    feature_list = []
                    loss = 0.0

                    for i,data in enumerate(test_data):
                        net.eval()
                        print("正在提取第{}个batch特征".format(i))
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs, features = net(images)
                        #print('features:',features)
                        feature_list.append(features)
                        #print('features:',features)
                        loss = criterion(outputs, labels)
                        preValue, predicted = torch.max(outputs.data, 1)
                        print('predicted label:',predicted)
                        #print("preValue:{},prediction:{}".format(preValue, predicted))
                        total += labels.size(0)
                        correct += torch.sum(predicted == labels.data)
                        for i in predicted:
                            predited_list.append(str(i.item()) + '\n')
                        for i in labels.data:
                            labels_list.append(str(i.item()) + '\n')
                        for i in outputs.cpu().data.numpy():
                            preValue_list.append(i)
                    #print('outputs:',outputs)
                    #print('preValue_list:',preValue_list)
                    acc = 100. * correct / total
                    f_loss1.write(str(loss.item()) + '\n')
                    f_acc1.write(str(float(acc)) + '\n')
                    print('测试分类准确率为:{:.4f}%, time:{}'.format(acc.item(), time.time() - since))

                    if round(acc.item(), 3) >= Best_ACCURACY :
                        Best_ACCURACY  = round(acc.item(), 3)
                        if not os.path.exists('./model/' + model_name + '/'+str(epoch)+'_'+str(Best_ACCURACY)):
                            os.makedirs('./model/' + model_name + '/'+str(epoch)+'_'+str(Best_ACCURACY))
                        torch.save(net, './model/' + model_name + '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/net.pkl')
                        tsne_features = torch.cat(feature_list, 0).cpu().data.numpy()
                        np.save('./model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/testing_features.npy',
                                tsne_features)  # 用于tsne
                        #print('tsne_features:',tsne_features)
                        np.save('./model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/preValue.npy', preValue_list)  # ROC 预测的概率值
                        save_file('./model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/testing_predicted.txt',
                                  predited_list)  # 预测label confusion matrix
                        save_file('./model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/testing_labels.txt',
                                  labels_list)  # 真实label confusion matrix
                        torch.save(net, './model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/net_best.pkl')

                        train_feature_list=[]
                        train_labels_list=[]
                        for i, data in enumerate(train_data, 0):
                            inputs, labels = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            # forward+backward
                            output,features = net(inputs)
                            train_feature_list.append(features)
                            for i in labels.data:
                                train_labels_list.append(str(i.item()) + '\n')
                        print(train_feature_list.__sizeof__())
                        train_features = torch.cat(train_feature_list, 0).cpu().data.numpy()
                         #tsne_features = torch.cat(feature_list, 0).cpu().data.numpy()
                        np.save('./model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/training_features.npy',
                                train_features)  # 用于tsne
                        save_file('./model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/training_labels.txt',
                                  train_labels_list)
                        #return
        #            if epoch+1 % 10 == 0:
        torch.save(net, './model/' + model_name +  '/'+str(epoch)+'_'+str(Best_ACCURACY)+'/net_end.pkl')
if __name__ == '__main__':
    main()









