import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from minas import minas, kmeans
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
#if it is run in a kernel:
%matplotlib qt 

n_update_plot = 1000
n_instances = 10000


def transform_y(y, y_tar):
    """
    Mutate predicted classes into the most plausible real classes.
    Inputs:
    - y: array of predicted labels
    - y_tar: array of target labels
    Output:
    - transformed predicted labels
    """
    new = np.array([0]*len(y))
    new[np.where(y==1)[0]] = 1
    new[np.where(y==2)[0]] = 2
    for i in set(y) - set([0,1,2]):
        # select y_target that corresponds to the y==i indices
        counts = np.unique(y_tar[np.where(y == i)[0]], return_counts=True)[1]
        # change to new class the y's that once belonged to class i 
        new[np.where(y==i)[0]] = counts.argmax(axis=0)+1
    return new

def CER(y, y_target):
    """
    Calculate Combiner Error measure.
    Input:
    - y: predicted classes
    - y_target: actual classes
    """
    soma = 0
    nr_y_target = np.unique(y_target, return_counts=True)
    ex = np.sum(nr_y_target[1])
    for i in set(y_target): 
        exCi = nr_y_target[1][np.where(nr_y_target[0]==i)[0]] 
        fpr = FPR(y, y_target, i)
        fnr = FNR(y, y_target, i)
        soma += (exCi / ex) * (fpr + fnr)
    return soma/2
    

def FPR(y, y_target, i):
    """
    Calculate FPR for class i.
    Input:
    - y: predicted classes
    - y_target: actual classes
    - i: class to calculate FPR
    """
    # FPR = FP / FP+TN
    fp = np.sum(y_target[np.where(y==i)[0]] != i)
    tn = np.sum(y_target[np.where(y!=i)[0]] != i)
    return fp/(fp+tn)


def FNR(y, y_target, i):
    """
    Calculate FNR for class i.
    Input:
    - y: predicted classes
    - y_target: actual classes
    - i: class to calculate FPR
    """
    fn = np.sum(y_target[np.where(y!=i)[0]] == i)
    tp = np.sum(y_target[np.where(y==i)[0]] == i)
    return fn/(fn+tp)

def UnkR(y, y_target):
    """
    Calculate Unknown Rate.
    Input:
    - y: predicted classes
    - y_target: actuall classes
    """
    soma = 0
    nr_y_target = np.unique(y_target, return_counts=True)
    for i in set(y_target):
        # nr. of istances of classe Ci that we're predicting as Unk
        unk = np.sum(y[np.where(y_target==i)[0]] == 0)
        exCi = nr_y_target[1][np.where(nr_y_target[0]==i)[0]] 
        soma += unk/exCi
    return soma/len(set(y_target))

def update_plot(cer, unk, novs, i):
    """
    Update plot.
    Input:
    - cer: Combined Error measure.
    - unk: UnkR measure
    - novs: list of timesteps denoting where there was detected a NP
    """
    try:
        plt.subplot(211)
        plt.plot(range(i-n_update_plot,i+1), cer, 'b-')
        for nov in novs:
            plt.axvline(nov, c='k')
        plt.subplot(212)
        plt.plot(range(i-n_update_plot,i+1),unk, 'r-')
        for nov in novs:
            plt.axvline(nov, c='k')
    except:
        plt.subplot(211)
        plt.plot(range(i-(n_update_plot-1),i+1),cer, 'b-')
        plt.subplot(212)
        plt.plot(range(i-(n_update_plot-1),i+1),unk, 'r-')
    plt.pause(0.1)



# 581012 rows, 55 cols
data = pd.read_csv('.\covtype.data', header = None)
# removing categorical variables for now.. we're losing a lot of info tho
data = data.drop(data.columns[10:54], axis=1)
# we don't shuffle the data as the paper says they do the initial training with the same original order

# train and stream classes same as the authors used in their experiments
train = data[data[54].isin([1,2])].iloc[:int(0.1*len(data))]
stream = data[data[54].isin([3,4,5,6,7])].append(data[data[54].isin([1,2])].iloc[int(0.1*len(data)):])
stream = stream.sample(frac=1, random_state=1)

# divide train and stream data into attributes and target
train_x = train.iloc[:,0:10].to_numpy()
train_y = train.iloc[:,10].to_numpy()
stream_x = stream.iloc[:,0:10].to_numpy()
stream_y = stream.iloc[:,10].to_numpy()

y_target = stream_y[:n_instances]

# build MINAS object and perform the initial training
m = minas(kmeans, True)
m.initial_training(train_x, train_y)

# lists of metrics
cer = list()
unkr = list()
novs = list()

fig = plt.figure()

# getting the data in a stream-wise fashion
for i,x in enumerate(stream_x[:n_instances]):
    m.online_phase(x)
    # record metrics after 1k instances
    if i > 1000:
        y = m.get_y()[1:]
        y = y.astype(int)
        y = transform_y(y, y_target[:len(y)])
        cer.append(CER(y[np.where(y!=0)], y_target[:len(y)][np.where(y!=0)]))
        unkr.append(UnkR(y, y_target[:len(y)]))
        if m.check_novelty():
            novs.append(i)
        # update plot at each new n_update_plot instances
        if i%n_update_plot == 0:
            update_plot(cer[-(n_update_plot+1):], unkr[-(n_update_plot+1):], novs, i)
            novs = list()

print("Novelty patterns detected: ", m.get_novelties()) 

cm = confusion_matrix(y_target=y_target, 
                    y_predicted=y, 
                    binary=False)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()
