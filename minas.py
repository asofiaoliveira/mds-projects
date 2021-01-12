import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import math

class minas:
    def __init__(self, clust_alg, evaluate = False):
        """
        Inputs:
        - clust_alg: the clustering algorithm to use in the model. 
        - its inputs have to be k, the number of clusters to form, and the train data, without target column
        """
        self.clust_alg = clust_alg
        self.t = 0
        self.evaluate = evaluate 
        self.y = np.array("remove")
        self.short_mem = pd.DataFrame(columns=["inst", "t", "y"])
        self.sleep_mem = list()
        self.novelties = 0
        self.novelty = False #whether new micro clusters were created in this time step

    def initial_training(self, train_x, train_y):
        """
        This function runs the initial training phase (offline) of the algorithm
        Inputs:
        - train_x: the "X" training data (without target labels) in the format of a numpy.array
        - train_y: an array with the target labels for each instance of X
        """
        self.classes = np.unique(train_y)
        self.model = list()
        # for each class in the training data, we create micro clusters associated with it
        for cl in self.classes:
            #the number of micro clusters is dependent on the number of instances,
            #so that a class with few instances doesn't end up with 100 clusters
            self.k = int(len(train_x[train_y==cl])/len(train_x)*100*len(self.classes))
            self.model += self.make_micro(instances = train_x[train_y==cl], cl = cl, t = 0)
        self.k = 100

    def get_model(self):
        """
        Model getter.
        Output:
        - self.model: list of active micro-clusters.
        """
        return self.model
    
    def get_novelties(self):
        """
        Getter for the number of novelties accumulator.
        Output:
        - self.novelties: number of novilties detected until now
        """
        return self.novelties

    def check_novelty(self):
        """
        Getter for the check novelty flag. This represents if there was a NP detected with the arrival of the last instance.
        Output:
        - self.novelty: boolean flag indicating the appearance of a new NP.
        """
        return self.novelty

    def get_shortmem(self):
        """
        Short Memory getter.
        Output:
        - self.short_mem: dataframe of alive instances classified as unknown (UNK)
        """
        return self.short_mem

    
    def get_sleepmem(self):
        """
        Sleep Memory getter.
        Output:
        - self.sleep_mem: list of asleep micro-clusters.
        """
        return self.sleep_mem

    
    def get_y(self):
        """
        Prediction getter.
        Output:
        - self.y: list of predicted classes to the instances of the incoming DS.
        """
        return self.y

    def online_phase(self, x):
        """
        This function is reponsible for receiving all new instances and processing them into a pre-existing cluster or the short-memory.
        It's also responsible for calling the novelty detection procedure when the short memory is lengthy enough.
        And for updating the active micro-clusters and remove old examples when the window size is reached.
        Input:
        - x: new instance
        """
        self.novelty = False
        # number of examples needed in short_mem to run novelty detection
        num_ex = 2000 
        # timesteps needed to check-up sleep-mem/old-examples checkup since last time
        window_size = 4000

        # update model time
        self.t += 1
        # find micro that it's closest to x
        dist, micro = self.closer_micro(x)
        # if x is close enough to the closest micro
        if(dist <= radius(self.model[micro], 1)):
            # update micro to include x
            self.update_micro(micro, x, self.t)
            if self.evaluate:
                # update self.y to include new prediction
                self.y = np.append(self.y,str(self.model[micro]["label"]))
        else:
            # move x to short_mem and classify it as UNK (for now)
            self.short_mem = self.short_mem.append({"inst": x, "t": self.t, "y": "UNK"}, ignore_index = True) 
            if self.evaluate:
                # update self.y to include new prediction
                self.y = np.append(self.y,"0")
            # if short_mem is lengthy enough, run the novelty detection procedure
            if(len(self.short_mem) > num_ex):
                self.novelty_detection()
        # if window_size instances have passed, remove instances older than window_size from short_mem 
        # and move micros that are not updated for longer than window_size to the sleepMem
        if(self.t % window_size == 0):
            self.move_sleepMem(window_size)
            self.remove_oldExamples(window_size)
    
    def novelty_detection(self):
        """
        Procedure responsible for checking if there are new micros in the short_mem and assigning them into an
        already existing class or a new class.
        """
        # number of minimum examples in a micro for it to be valid
        num_ex = len(self.short_mem)/self.k
        # discover new micros in the short_mem data
        tmp = self.make_micro(np.array(self.short_mem["inst"].values.tolist()), None, self.short_mem["t"])
        # for each new micro in the short_mem data
        for i, micro in enumerate(tmp):
            # find the closest micro in the active model
            d_closest, ind_closest = self.closest_micro(micro, self.model)
            # find the closest micro in the sleeping model
            dist, ind = self.closest_micro(micro, self.sleep_mem)
            # if the new micro is valid (cohesive with the closest active or sleeping micro and with greater than num_ex instances)
            if (self.cohesive(micro, ind_closest) or self.cohesive(micro, ind, True)) and micro['n'] > num_ex:
                thresh = radius(micro, 1.1)
                # if new micro is close to the active micro, assign it the same label as that active micro
                if d_closest <= thresh:
                    micro['label'] = self.model[ind_closest]['label']
                else:
                    # if new micro is close to the sleeping micro, assign it the same label as that sleeping micro
                    if dist <= thresh:
                        micro['label'] = self.sleep_mem[ind]['label']
                        self.remove_sleepMem(ind)
                    # if all else fails, then assign it a new not yet seen class
                    else:
                        micro['label'] = max(self.classes)+1
                        self.classes = np.append(self.classes,max(self.classes)+1)
                # get indices of instances to maintain
                ind = self.short_mem.y != "k="+str(i)
                if self.evaluate:
                    self.y[self.short_mem[~ind].t.to_numpy(dtype=int)] = micro["label"]
                # remove allocated instances from short_mem since they now belog to a micro
                self.short_mem = self.short_mem[ind]
                # append new micro to active model
                self.model.append(micro)
                # we found a NP, so set the novelty flag to true and increase the counter
                self.novelty = True
                self.novelties += 1

    def remove_sleepMem(self, ind):
        """
        Remove instances from the sleep_mem and put them back in the model
        Input:
        - Indices of clusters to remove from sleepMem
        """
        self.model.append(self.sleep_mem[ind])
        self.sleep_mem.pop(ind)

    def cohesive(self, micro, ind, sleep=False):
        """
        Function to verify if a given micro-cluster is cohesive.
        Input:
        - micro: micro-cluster to verify cohisiveness
        - ind: index of closest micro-cluster
        - sleep: boolean flag indicating if closest micro-cluster is in the sleepMem
        Output:
        - Boolean value indicating if micro is cohesive or not
        """
        if sleep == False:
            closest = self.model[ind]
        else:
            if ind == None: return False
            closest = self.sleep_mem[ind]
        centr = centroid(micro)
        b = distance(centr, centroid(closest))
        a = radius(micro, 1)
        return (b-a)/max(b, a) > 0

    def make_micro(self, instances, cl, t):
        """
        This function actually creates the model given instances. 
        It runs the clustering algorithm and then calculates the linear sum, square sum, number of points and time of each micro cluster
        Input:
        - instances: array of instances to process
        - cl: class to assign the new micro-cluster(s)
        - t: time step of MINAS
        Output:
        - model: updated model with new micro_cluster
        """
        tmp = self.clust_alg(self.k, instances)
        model = list()
        for j in range(0,self.k):
            ind = np.where(tmp.labels_ == j)[0]
            inst = instances[ind, :]
            try:
                t_max = max(t[ind].values)
                self.short_mem.iloc[ind,2] = "k="+str(j)
            except:
                t_max = t
            n = len(inst)
            ls = np.sum(inst, axis = 0)
            ss = np.sum(inst**2, axis = 0)
            model.append(dict(n=n, ls=ls, ss=ss, label=cl, t=t_max)) 
        return model

    def closest_micro(self, micro, l):
        """
        Discover the which micro-cluster of list l is closer to micro-cluster micro.
        Input:
        - micro: subject micro-cluster
        - l: list of micro-clusters in which to find the closest micro-cluster of micro
        Output:
        - dmin: distance to closest micro-cluster
        - ind: index in l of closest micro-cluster
        """
        cent = centroid(micro)
        d_min = math.inf
        ind = None
        if len(l)>0:
            for i, m in enumerate(l):
                d = distance(centroid(m), cent)
                if d < d_min:
                    d_min = d
                    ind = i
        return d_min, ind

    def move_sleepMem(self, window_size):
        """
        This functions serves the purpose of moving micro-clusters that didn't receive any
        instances in the last window_size timesteps to the sleep_mem, since those micros don't seem to represent the DS anymore.
        Input:
        - window_size: time needed since last update to the micro to move it to sleepMem
        """
        to_sleep = np.where((self.t - np.array([d['t'] for d in self.model])) > window_size)[0]
        if len(to_sleep)>0:
            self.sleep_mem += list(self.model[i] for i in to_sleep)
            for i in reversed(to_sleep):
                self.model.pop(i)

    def remove_oldExamples(self, window_size):
        """
        This function serves the purpose of removing instances in the shortMem that 
        have not been used and are older than window_size timestes.
        Input:
        - window_size: value that indicates how old can an instance be
        """
        for i in reversed(self.short_mem.index.to_list()):
            if((self.t - self.short_mem.loc[i,'t']) > window_size): 
                self.short_mem.drop(i) 

    def update_micro(self, micro, instance, t):
        """
        Function that adds a new instance to an already existing micro.
        Input:
        - micro: micro to add the instance to
        - instance: instance to add to the micro
        - t: MINAS timestep at which this is happening
        """
        self.model[micro]['n'] += 1
        self.model[micro]['ls'] += instance
        self.model[micro]['ss'] += instance**2
        self.model[micro]['t'] = t

    def closer_micro(self, instance):
        """
        Discover which micro-cluster in model are closer to instance.
        Input:
        - instance: subject instance
        Output:
        - best_dist: distance to the closest micro-cluster in model
        - ind: index of the closest micro-cluster in model
        """
        best_dist = math.inf
        for i in range(len(self.model)):
            dist = distance(instance, centroid(self.model[i]))
            if(dist < best_dist):
                best_dist = dist
                ind = i
        return best_dist, ind

def radius(micro, f = 1):
    """
    Calculate radius of a micro. The radius is given by the standard deviation of the distance between the examples
    and the centroid, multiplied by a factor f.
    Input:
    - micro: micro to calculate the radius
    - f: factor to multiply by the radius
    Output:
    - Radius of the micro-cluster
    """
    return f*np.sqrt(np.sum((micro['ss'] - 2*micro['ls']*centroid(micro) + micro['n']*(centroid(micro)**2))/micro['n']))

def centroid(micro):
    """
    Calculate centroid of a micro
    Input:
    - micro: micro-cluster to calculate the centroid of
    Output:
    - Centroid coordinates
    """
    return micro['ls']/micro['n']

def distance(p1, p2):
    """
    Calculate Euclidean distance between two points
    Input:
    - p1: point 1
    - p2: point 2
    Output:
    - Distance between the two points.
    """
    return np.sqrt(np.sum((p1-p2)**2, axis = 0)) # axis = 0? irrelevante acho, só tem 1 eixo


def kmeans(k, data):
    """
    Simple KMeans clustring.
    Input:
    - k: number of clusters to use in the KMeans procedure
    - data: data to do the KMeans fit
    """
    return KMeans(n_clusters=k, random_state=0).fit(data)