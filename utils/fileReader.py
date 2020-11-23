import re

def get_list(file_path):
    target_list=[]
    f=open(file_path,'r')
    for line in f:
        target_list.append(line.strip())
    return target_list

if __name__ == '__main__':

    list=get_list('../model/DR30/con_labels.txt')
    print(list)
    for i in range(5):
        print(i)