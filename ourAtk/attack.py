import scipy
import scipy.sparse as sp
import numpy as np
import time
import tensorflow as tf
from ourAtk.utils import *
from ourAtk.data import *
from gcnMaster.gcn import train


flags = tf.app.flags
FLAGS = flags.FLAGS


def sigmoid(x):
    '''sigmoid for -1/+1'''
    return np.exp(x) / (np.exp(x) + np.exp(-x))


class Attack(object):
    def __init__(self, dataName, gpu_id,data_setting,atkEpoch,gcnL2):
        self.dataName = dataName
        self.setting = data_setting
        self.data = Data(self.dataName,self.setting)
        self.gpu_id = gpu_id
        self.atkEpoch = atkEpoch
        self.gcnL2 = gcnL2
    
    def set_c(self, c_max):
        # attack budget
        self.c_max = c_max

    def get_tau(self,c_max):
        # select the hyperparam tau (larger tau, the closer to the step function)
        atk_list = []
        tau_list = [1,2,4,8,16,32,64,128]
        maxTau = 0
        maxAcc = 100000
        maxOuts = None
        for tau in tau_list:
            self.tau = tau
            outs = self.gradient_attack_sgd(c_max)
            acc_atk = outs[3]
            atk_list.append(acc_atk)
            if acc_atk <= maxAcc:
                maxAcc = acc_atk
                maxTau = tau
                maxOuts = outs
        self.tau = maxTau
        # print("atk_list under all tau:",atk_list," final tau:",self.tau," final acc:",maxAcc)
        return maxOuts
               
                
                
    def GCN_test(self):
        y_train = np.zeros(self.data._Z_obs.shape)
        y_val = np.zeros(self.data._Z_obs.shape)
        y_test = np.zeros(self.data._Z_obs.shape)
        y_train[self.data.train_mask, :] = self.data._Z_obs[self.data.train_mask, :]
        y_val[self.data.val_mask, :] = self.data._Z_obs[self.data.val_mask, :]
        y_test[self.data.test_mask, :] = self.data._Z_obs[self.data.test_mask, :]
        assert self.data._z_obs.shape[0] == self.data._Z_obs.shape[0],"the shape of _z_obs and _Z_obs are not equal!"
        acc, eachAcc, predProbs = train.GCN_Train(self.data._A_obs, self.data._X_obs,\
                            y_train, y_val, y_test, self.data.train_mask, self.data.val_mask, self.data.test_mask)
        return acc
        
    def closedForm_bin(self,trainLabels):
        # closed form of GCN
        y_pred = np.dot(self.K,trainLabels) #dot(K,y_L)
        mse_org = accuracy(np.sign(y_pred), self.data._z_obs[self.data.split_unlabeled])
        return mse_org, y_pred
        
    def getK_GCN(self):
        A = preprocess_graph(self.data._A_obs[self.data.nodes_AB_all,:][:,self.data.nodes_AB_all]).tocsr()#1.add self-loop 2.sys-normalize
        X = row_normalize(self.data._X_obs[self.data.nodes_AB_all]).tocsr()
        X_bar = A.dot(A).dot(X).tocsr()
        X_bar_l = X_bar[self.data.split_train,:]
        tmp = X_bar_l.T.dot(X_bar_l)
        tmp = sp.csr_matrix(scipy.linalg.pinv(tmp.toarray()+ self.gcnL2*np.identity(tmp.shape[0])))
        tmp = tmp.dot(X_bar_l.T)
        K = X_bar.dot(tmp)[self.data.split_unlabeled,:].toarray()
        return K

        
    def gradient_attack_sgd(self,c_max):
        _z_obs_org = np.array(self.data._z_obs)# _z_obs_org saves the clean labels
        self.data._z_obs = BinaryLabelToPosNeg(self.data._z_obs)# turn labels form {0,1} to {-1,1}
        y_l = np.reshape(self.data._z_obs[self.data.split_train],(len(self.data.split_train),1))
        y_u = np.reshape(self.data._z_obs[self.data.split_unlabeled],(len(self.data.split_unlabeled),1))
        tf.reset_default_graph()
        alpha = tf.Variable(name='alpha', initial_value=(0.5 * np.ones_like(y_l)),dtype='float32')# the parameter of Bernoulli
        # reparameterization trick for sampling
        epsilon = tf.placeholder(tf.float32,name='epsilon')
        tmp = tf.exp((tf.log(alpha / (tf.constant(1.0,dtype='float32') - alpha)) + epsilon) / tf.constant(0.5))
        z = tf.constant(2.0,dtype='float32') / (tf.constant(1.0,dtype='float32') + tmp) - tf.constant(1.0,dtype='float32')     # normalize z from [0, 1] to [-1, 1]
        y_l_tmp = tf.constant(y_l,dtype='float32') * z
        # approximated closed form of GCN
        y_u_preds = tf.nn.tanh(self.tau*tf.matmul(tf.constant(self.K,dtype='float32'),y_l_tmp)) 
        loss = tf.reduce_mean(y_u_preds*tf.constant(y_u,dtype='float32'))
        optimizer = tf.train.GradientDescentOptimizer(1.0e-4)
        opt_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            for step in range(self.atkEpoch):
                ep = np.random.gumbel(len(y_l)) - np.random.gumbel(len(y_l))# reparameterization trick
                sess.run(opt_op,feed_dict={epsilon: ep})# update alpha
            alpha = sess.run(alpha,feed_dict={epsilon: ep})# return alpha
        alpha = np.reshape(np.array(alpha),(len(self.data.split_train)))
        idx = np.argsort(alpha)[::-1]
        d_y = np.ones_like(self.data._z_obs[self.data.split_train])
        count = 0
        flip = {}
        for i in idx:
            if alpha[i] > 0.5 and count < c_max:
                d_y[i] = -1
                count += 1
                flip[self.data.split_train[i]] = alpha[i]
                if count == c_max:
                    break
        trainLabels = self.data._z_obs[self.data.split_train]
        acc_cln,pred_closed_cln = self.closedForm_bin(trainLabels)
        trainLabels_atk = np.array(self.data._z_obs)
        trainLabels_atk[list(flip.keys())] = self.data._z_obs[list(flip.keys())] *(-1)
        trainLabels_atk = trainLabels_atk[self.data.split_train]
        acc_atk,pred_closed_atk = self.closedForm_bin(trainLabels_atk)
        # print(' #flip: ', flip, '  clean acc:',acc_cln,' atk acc:', acc_atk)
        _z_obs_atk = np.array(self.data._z_obs)
        _z_obs_atk[self.data.split_train] = _z_obs_atk[self.data.split_train] * d_y
        self.data._z_obs = _z_obs_org
        return d_y,BinaryLabelTo01(_z_obs_atk),acc_cln,acc_atk,pred_closed_cln,pred_closed_atk#outs 不要随便改动
    
    def binaryAttack_multiclass(self,c_max,a=2,b=3):
        #1. change multi-class to binary with the selected classes a and b
        time1 = time.clock()
        self.data.resetBinaryClass_init(a,b)
        #2. get K
        self.K = self.getK_GCN()
        time2 = time.clock()
        #3.get flip nodes and best hyperparameter tau
        d_y,_,acc_bin_cln,acc_bin_atk,preds_closed_cln,preds_closed_atk= self.get_tau(c_max)
        flipNodes = np.array(self.data.nodes_AB_train)[d_y!=1]
        # print("flipped labels:",flipNodes)
        time3 = time.clock()
        print("overall:",time3 -time1,"pre-process:",time2-time1, "optimize:",time3-time2)            
        #4. change the perturbed binary labels back to multi-class labels
        self.data.recover_data()
        _Z_obs_atk = np.array(self.data._Z_obs)
        for i in flipNodes:
            if i not in self.data.nodes_AB_train.tolist():
                print("wrong flipNodes!")
                exit()
            if self.data._z_obs[i] == a:
                _Z_obs_atk[i][a] = 0
                _Z_obs_atk[i][b] = 1
            elif self.data._z_obs[i] == b:
                _Z_obs_atk[i][a] = 1
                _Z_obs_atk[i][b] = 0
            else:
                print("wrong flipNodes!")
                exit()
        self.data._Z_obs = _Z_obs_atk
        self.data.resetz_by_Z()
        
        # #5. attack multi-class classification
        # acc_mul_atk = self.GCN_test()
        # self.data.recover_data()
        
        # #6. clean  multi-class classification
        # acc_mul_cln = self.GCN_test()

        # print("flip:"+str(flipNodes)+"\n"+\
            # "tau:"+str(self.tau)+"\t"+\
            # "acc of binary classification (clean):"+str(acc_bin_cln)+"\n"+\
            # "acc of binary classification (attack):"+str(acc_bin_atk)+"\n"+\
            # "acc of multi-class classification (clean):"+str(acc_mul_cln)+"\n"+\
            # "acc of multi-class classification (attack):"+str(acc_mul_atk)+"\n")
            
        acc_test_mul_atk = []    
        for i in range(5):
            acc = self.GCN_test()
            acc_test_mul_atk.append(acc)
        self.data.recover_data()
        print(acc_test_mul_atk)
        acc_mul = sum(acc_test_mul_atk)*1.0/len(acc_test_mul_atk)
        print("overall acc_test_mul_atk:"+str(acc_mul))
        return acc_mul
