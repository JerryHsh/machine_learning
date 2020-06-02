import numpy as np

import operator as op


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def myfirstknn(exp,dataset,labels,k):
    datasize=dataset.shape[0]
    classcount={}
    if k>datasize:
        return classcount,'0'
    cal_matrix=np.tile(exp,(datasize,1))
    diff_matrix=cal_matrix-dataset
    sq_diff_matrix=diff_matrix**2
    sq_distance=sq_diff_matrix.sum(axis=1)
    distance=sq_distance**0.5
    sorted_distanceindices=distance.argsort()
    for i in range(k):
        current_label=labels[sorted_distanceindices[i]]
        classcount[current_label]=classcount.get(current_label,0)+1
    rec_list=sorted(classcount.items(),key=op.itemgetter(1),reverse=True)
    return rec_list,rec_list[0][0]

group,labels=create_data_set()
rec_list,result=myfirstknn([0,0],group,labels,3)
print(result)