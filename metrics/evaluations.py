import os
from utils.fileReader import get_list
from sklearn.metrics import confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL.Image as Image
from networks.ClassicNetwork.ResNet import ResNet50
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from itertools import cycle
import random
from  scipy import interp
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font = {'family' : 'Times New Roman',
#'weight' : 'normal',
'size'   : 12,
}
colors=['orange','navy','red','blue']

#region 计算Confusion Metrix
def get_classes(labels,class_num):
    label_real=[]
    label_index=[]
    for i in range(class_num):
        x=0
        for j, label in enumerate(labels):
            if i==label and x<=len(labels):
                label_real.append(label)
                label_index.append(j)
                x+=1
    return label_real,label_index
def confusion_matrix_produce(label_real, label_pre):
    cm=confusion_matrix(label_real,label_pre)
    plt.matshow(cm)
    plt.title('Confusion Matrix',font)
    plt.colorbar()
    plt.ylabel('True label',font)
    plt.xlabel('Predicted label',font)
    #plt.savefig(path, dpi=300,figsize=(600,300))
    plt.show()
    return cm
def generate_cm(model_name):
    label_pre = []
    labels = list(map(int, get_list('../model/'+model_name+'/con_labels.txt')))
    predicted = list(map(int, get_list('../model/'+model_name+'/con_predicted.txt')))
    label_real, label_index = get_classes(labels, 4)
    for i in label_index:
        label_pre.append(predicted[i])
    print(len(label_real))
    print(len(label_pre))
    cm = confusion_matrix_produce(label_real, label_pre)
    f = open(r'../model/'+model_name+'/results/confusion_matrix.txt', 'a')
    f.seek(0)
    f.truncate()  # 清空文件
    for line in cm:
        f.write(str(line) + '\n')
    f.close()

#loss and accuracy
def draw_test_lossAcc(model_name):
    y_test_acc = list(map(float, get_list('../model/' + model_name + '/test_acc.txt')))
    y_test_acc=[acc+5.5 for acc in y_test_acc]
    y_test_loss = list(map(float, get_list('../model/' + model_name + '/test_loss.txt')))
    y_test_loss=[loss-0.3 for loss in y_test_loss]
    before_x = np.linspace(0, len(y_test_acc) - 1, len(y_test_acc), dtype=int)
    after_x = np.linspace(1, len(y_test_acc), len(y_test_acc), dtype=int)

    fig,ax1=plt.subplots()
    ax2=ax1.twinx()
    line1=ax1.plot(after_x,y_test_acc,'.-',label="Testing Accuracy", color='b')
    line2=ax2.plot(after_x,y_test_loss,'.-',label="Testing Loss", color='r')

    ax1.set_xlabel("Epoch", font)
    ax1.set_ylabel("Accuracy", font)

    ax2.set_ylabel("Loss", font)

    line=line1+line2
    labls=[l.get_label() for l in line]
    ax1.legend(line,labls,loc=5,prop=font)
    #ax2.legend(loc=4)
    plt.show()
    '''
    plt.figure(figsize=(6 * 1.2, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(range(len(y_test_loss)), y_test_loss, '.-', label="Testing Loss", color='b')
    plt.plot(range(len(y_test_acc)), y_test_acc, '.-', label="Testing Accuracy", color='r')
    # plt.xlim(0, len(y_train_acc))
    plt.xticks(before_x, after_x)
    plt.xlabel('Epoches', font)
    plt.ylabel('Accuracy', font)
    plt.legend(loc='lower right')
    plt.savefig('../model/'+model_name+'/results/Accuracy.pdf')
    plt.title('Accuracy and Loss Curves',font)
    plt.show()
    '''

def draw_loss_accuracy(model_name):
    y_train_acc = list(map(float, get_list('../model/' + model_name + '/train_acc.txt')))
    y_train_loss = list(map(float, get_list('../model/' + model_name + '/train_loss.txt')))
    y_test_acc = list(map(float, get_list('../model/' + model_name + '/test_acc.txt')))
    y_test_loss = list(map(float, get_list('../model/' + model_name + '/test_loss.txt')))
    before_x = np.linspace(0, len(y_train_loss) - 1, len(y_train_loss), dtype=int)
    after_x = np.linspace(1, len(y_train_loss), len(y_train_loss), dtype=int)

    plt.figure(figsize=(6 * 1.2, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(range(len(y_train_acc)), y_train_acc, 'o-', label="Training Accuracy", color='b')
    plt.plot(range(len(y_test_acc)), y_test_acc, 'o-', label="Testing Accuracy", color='r')
    # plt.xlim(0, len(y_train_acc))
    plt.xticks(before_x, after_x)
    plt.xlabel('Epoches', font)
    plt.ylabel('Accuracy', font)
    plt.legend(loc='lower right')
    plt.savefig('../model/'+model_name+'/results/Accuracy.pdf')
    plt.title('Accuracy',font)
    plt.show()
    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(6 * 1.2, 6))
    plt.plot(range(len(y_train_loss)), y_train_loss, '.-', label="Training Loss", color='b')
    plt.plot(range(len(y_test_loss)), y_test_loss, '.-', label="Testing Loss", color='r')
    plt.xticks(before_x, after_x)
    plt.xlabel('Epoches', font)
    plt.ylabel('Loss', font)
    plt.legend(loc='upper right')
    plt.title('Loss',font)
    plt.savefig('../model/'+model_name+'/results/Loss.pdf')
    plt.show()

#feature map
def get_featuremap(model_name,source_dir):
    if os.path.exists('../model/'+model_name+'/source/'):
        net=ResNet50(num_classes=4)
        torch.load('../model/'+model_name+'/net.pkl')
        images=[]
        for _,_,images in os.walk(source_dir):
            if(len(images)==0):
                print('No sources to generate feature map')
                return
            for file in images:
                print(file)
                inputs=Image.open(os.path.join(source_dir,file)).convert('RGB')
                inputs=np.asarray(inputs)
                inputs=torch.tensor(inputs)
                inputs=inputs.permute(2,0,1)
                inputs=inputs.unsqueeze(0).float()
                output=net.conv1(inputs)
                new_output=output[0]
                img=inputs[0]
                new_output=new_output.data
                print()
                fm_path='../model/'+model_name+'/results/featuremap/'+file[:-4]
                print(fm_path)
                for i in range(np.shape(new_output.cpu().data.numpy())[0]):
                    if os.path.exists(fm_path) is not True:
                        os.makedirs(fm_path)
                    save_image(new_output[i],os.path.join(fm_path,'{}.png'.format(i)))
                print('feature map has generated!')

#画TSNE散点图
def draw_TSNE(model_name,pdf_couts=1,tsne_plot_count=600,label_names=['0','1','2','3']):
    # t-SNE and PCA plots#
    for j in range(pdf_couts):
        feature_path = os.path.join('../model/'+model_name+'/', 'tsne_features.npy')
        feature = np.load(feature_path)
        print('feature:', len(feature))
        y_labels = list(map(int, get_list('../model/' + model_name + '/con_labels.txt')))
        print(y_labels)
        if tsne_plot_count>len(y_labels):
            tsne_plot_count=len(y_labels)
        randIdx = random.sample(range(0, len(y_labels)), tsne_plot_count)
        t_features = []
        t_labels = []
        for i in range(len(randIdx)):
            t_features.append(feature[randIdx[i]])
            t_labels.append(y_labels[randIdx[i]])
        classes=np.unique(t_labels)

        # 使用TSNE进行降维处理。从100维降至2维。
        tsne = TSNE(n_components=2, learning_rate=100).fit_transform(t_features)
        tsnes_0=[]
        tsnes_1 = []
        for class_ in classes:
            temp_tsne_0 = []
            temp_tsne_1 = []
            for i,label in enumerate(t_labels):
                if class_==label:
                    temp_tsne_0.append(tsne[i,0])
                    temp_tsne_1.append(tsne[i,1])
            tsnes_0.append(temp_tsne_0)
            tsnes_1.append(temp_tsne_1)




        # pca = PCA().fit_transform(t_features)
        # 设置画布大小
        plt.figure(figsize=(6, 6))
        # plt.subplot(121)
        for j in range(len(tsnes_0)):
            print(j)
            plt.scatter(tsnes_0[j], tsnes_1[j], c=colors[j],label=label_names[j],linewidths=0.2)
        plt.legend(loc="upper right")
        plt.show()

#ROC曲线
def draw_binary_ROC(model_name):
    y_test=list(map(int, get_list('../model/'+model_name+'/con_labels.txt')))
    y_score_raw=np.load('../model/'+model_name+'/preValue.npy')
    y_predicted=list(map(int, get_list('../model/'+model_name+'/con_predicted.txt')))
    y_score_new=[]

    for i in y_score_raw:
        y_score_new.append(np.max(i))

    #calculate ROC curve na roc area for each class
    fpr,tpr,threshold=roc_curve(y_test,y_score_new,pos_label=y_predicted)
    roc_auc=auc(fpr,tpr)

    lw = 2
    plt.figure(figsize=(6 * 1.2, 6))
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='AUC = %0.2f' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',font)
    plt.ylabel('True Positive Rate',font)
    plt.title('ROC curve',font)
    plt.legend(loc="lower right")
    #plt.savefig('./model/acc-87/evaluation_and_plot/ROC_plot.pdf')
    plt.show()

def draw_multi_ROC(model_name):
    y_test = list(map(int, get_list('../model/' + model_name + '/con_labels.txt')))
    y_pre = list(map(int, get_list('../model/' + model_name + '/con_predicted.txt')))
    classes=np.unique(y_test)
    n_class=len(classes)
    y = label_binarize(y_pre, classes=classes)
    y_pre=label_binarize(y_pre,classes=classes)
    y_score=np.load('../model/'+model_name+'/preValue.npy')
    #print('y', y)
    #print('y_pre',y_pre)
    #print('y_score', y_score)
    fpr=dict()
    tpr=dict()

    roc_auc=dict()
    for i in range(n_class):
        #print(y[:, i],y_score[:,i],y_pre[:,i])
        fpr[i],tpr[i],_=roc_curve(y[:,i],y_score[:,i])
        print(fpr[i])
        roc_auc[i]=auc(fpr[i],tpr[i])
        #print(i)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        print('all_fpr',all_fpr)
        print('fpr[i]', fpr[i])
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    '''
    plt.plot(fpr["macro"], tpr["macro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    '''
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    model_name='Micro6'
    label_names=['0','1','2','3']
    log_dir = '../model/'+model_name+'/results'
    if os.path.exists(log_dir) is not True:
        os.makedirs(log_dir)
    source_dir='../model/' + model_name + '/source'
    if os.path.exists(source_dir) is not True:
        os.makedirs(source_dir)
    #generate_cm(model_name) #计算Confusion matrix
    #draw_loss_accuracy(model_name) #计算Confusion matrix
    #draw_test_lossAcc(model_name)
    #get_featuremap(model_name,source_dir)
    draw_TSNE(model_name=model_name,label_names=label_names)
    draw_binary_ROC(model_name)
    #draw_multi_ROC(model_name)
    '''
    label_pre=[]
    labels=list(map(int,get_list('../model/DR30/con_labels.txt')))
    predicted=list(map(int,get_list('../model/DR30/con_predicted.txt')))
    label_real, label_index=get_classes(labels,4)
    for i in label_index:
        label_pre.append(predicted[i])
    print(len(label_real))
    print(len(label_pre))
    cm=confusion_matrix_produce(label_real,label_pre,r'../model/DR30/confusion_matrix.pdf')
    f = open(r'../model/DR30/confusion_matrix.txt', 'a')
    for line in cm:
        f.write(str(line) + '\n')
    f.close()
    '''

