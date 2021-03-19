import numpy as np
from ourAtk.utils import *
from ourAtk.loadData import *


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

class Data():
    def __init__(self, dataName, setting):
        self.dataName = dataName
        self.setting = setting
        self.drdata = Dataset(root='/home/zmm/advGraph/nettack-master/data/', name=self.dataName, setting=self.setting)
        self._A_obs, self._X_obs, self._z_obs_org = self.drdata.adj,  self.drdata.features,  self.drdata.labels
        assert self._A_obs.shape[0] == self._z_obs_org.shape[0], "dimension not match!"
        self._K = self._z_obs_org.max()+1
        self._z_obs = np.array(self._z_obs_org)
        self.resetZ_by_z()
        self._Z_obs_org = np.array(self._Z_obs)        
        self._N = self._A_obs.shape[0]
        self._An = preprocess_graph(self._A_obs)#加自环，归一化
        self.degrees = self._A_obs.sum(0).A1 #度向量
        self.seed = 15
        np.random.seed(self.seed)
        
        self.sizes = [16, self._K] #16*label个数,16是hidden size
        
        assert self._z_obs_org.shape[0] == self._Z_obs_org.nonzero()[0].shape[0],"_Z_obs have empty label!"
    
    def split_data(self):
        self.split_train, self.split_val, self.split_test = self.drdata.get_train_val_test()
        self.split_unlabeled = [ i for i in range(self._A_obs.shape[0]) if i not in self.split_train]
        self.train_mask = sample_mask(self.split_train, self._A_obs.shape[0])
        self.val_mask = sample_mask(self.split_val, self._A_obs.shape[0])
        self.test_mask = sample_mask(self.split_test, self._A_obs.shape[0])
        self.test_idx = [self.split_unlabeled.index(x) for x in self.split_test]#test在unlabel里的下标
        self.split_train_org, self.split_val_org, self.split_test_org, self.split_unlabeled_org = list(self.split_train), list(self.split_val), list(self.split_test), list(self.split_unlabeled)
        
    def recover_data(self):
        self._z_obs = np.array(self._z_obs_org)
        self._Z_obs = np.array(self._Z_obs_org)
        self.split_train = list(self.split_train_org)
        self.split_val = list(self.split_val_org)
        self.split_test = list(self.split_test_org)
        self.split_unlabeled = list(self.split_unlabeled_org)
        self.train_mask = sample_mask(self.split_train, self._A_obs.shape[0])
        self.val_mask = sample_mask(self.split_val, self._A_obs.shape[0])
        self.test_mask = sample_mask(self.split_test, self._A_obs.shape[0])
        self.test_idx = [self.split_unlabeled.index(x) for x in self.split_test]#test在unlabel里的下标

    
    def getMaxAB(self):
        count = []
        for i in range(self._K):
            count.append(self._z_obs[self._z_obs == i].shape[0])
        # print(count)
        a = count.index(max(count))
        count[count.index(max(count))]=0
        b = count.index(max(count))
        return a,b
       

    def resetBinaryClass_init(self,a=2,b=3):
        self.a = a
        self.b = b
        if self._K == 2:
            self.nodes_AB_all = range(self._z_obs.shape[0])
            self.nodes_AB_train = np.array(self.split_train)
            return
        nodes_labelA = np.squeeze(np.argwhere(self._z_obs == self.a))
        nodes_labelB = np.squeeze(np.argwhere(self._z_obs == self.b))
        nodes_labelA_train = get_intersection(nodes_labelA,self.split_train)
        nodes_labelB_train = get_intersection(nodes_labelB,self.split_train)
        nodes_labelA_un = get_intersection(nodes_labelA,self.split_unlabeled)
        nodes_labelB_un = get_intersection(nodes_labelB,self.split_unlabeled)
        nodes_labelA_test = get_intersection(nodes_labelA,self.split_test)
        nodes_labelB_test = get_intersection(nodes_labelB,self.split_test)
        nodes_labelA_val = get_intersection(nodes_labelA,self.split_val)
        nodes_labelB_val = get_intersection(nodes_labelB,self.split_val)
        self.nodes_AB_train = np.hstack((nodes_labelA_train, nodes_labelB_train))#训练集的node id
        nodes_AB_test = np.hstack((nodes_labelA_test, nodes_labelB_test))#test的node id
        nodes_AB_val = np.hstack((nodes_labelA_val, nodes_labelB_val))#valid 的node id
        nodes_AB_un = np.hstack((nodes_labelA_un, nodes_labelB_un))#unlabel的node id
        self.nodes_AB_all = np.hstack((self.nodes_AB_train, nodes_AB_un))#all a b 的node id
        self._z_obs = self._z_obs[self.nodes_AB_all]
        self._z_obs[self._z_obs == a] = 0
        self._z_obs[self._z_obs == b] = 1
        self._Z_obs = np.zeros((len(self.nodes_AB_all),self._K))
        self._Z_obs[np.arange(len(self.nodes_AB_all)), np.array(self._z_obs)] = 1
        seed = 15
        np.random.seed(seed)
        self.sizes = [16, self._K] #16*label个数
        newIdx = self.nodes_AB_all.tolist()
        self.split_train = [newIdx.index(x) for x in self.nodes_AB_train]
        self.split_test = [newIdx.index(x) for x in nodes_AB_test]
        self.split_val = [newIdx.index(x) for x in nodes_AB_val]
        self.split_unlabeled = [newIdx.index(x) for x in nodes_AB_un]
        self.test_idx = [self.split_unlabeled.index(x) for x in self.split_test]#test在unlabel里的下标
        self.train_mask = sample_mask(self.split_train, self._A_obs.shape[0])
        self.val_mask = sample_mask(self.split_val, self._A_obs.shape[0])
        self.test_mask = sample_mask(self.split_test, self._A_obs.shape[0])
        
    def resetZ_by_z(self):
        self._Z_obs = np.zeros((self._z_obs.shape[0],self._K))
        self._Z_obs[np.arange(self._z_obs.shape[0]), np.array(self._z_obs)] = 1
        
    def resetz_by_Z(self):
        self._z_obs = self._Z_obs.nonzero()[1]
        