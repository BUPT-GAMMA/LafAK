import random
import pickle as pkl
import tensorflow as tf
# import attack
from ourAtk import attack
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)  


gpu_id = '1'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


dataName = 'citeseer' # 'cora','citeseer','pubmed'
c_max = 4 # attack budget=4 (namely flip rate=0.2: the labels of two class equals to 40 in )

atk = attack.Attack(dataName, gpu_id,'gcn',100,5e-3)
for i_split in range(5): #We randomly split each dataset for five times        
    # 1.use the splitted data:
    with open("data/"+dataName+"_"+str(i_split),'rb') as f:
        atk.data = pkl.load(f, encoding='latin1')
    
    # 2.select the two largest classes for attack
    class_a, class_b = atk.data.getMaxAB()
    print("the two largest classes for attack:", class_a, class_b)

    atk.binaryAttack_multiclass(c_max, class_a, class_b)