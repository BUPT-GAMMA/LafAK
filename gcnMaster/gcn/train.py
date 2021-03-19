from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcnMaster.gcn.utils import *
from gcnMaster.gcn.models import GCN, MLP



def GCN_Train(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask):
    tf.set_random_seed(123)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # Some preprocessing
    features = preprocess_features(features)
    support = [preprocess_adj(adj)]
    num_supports = 1
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # Define model evaluation function
    def evaluate(sess, features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.eachAccuracy,model.preds], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    trainLoss = []
    valLoss = []
    train_acc = []
    val_acc = []
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, eachAcc,_, duration = evaluate(sess, features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        
        

        # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              # "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              # "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            # print("Early stopping...")
            break
        
        trainLoss.append(outs[1])
        train_acc.append(outs[2])
        valLoss.append(cost)
        val_acc.append(acc)

    # Testing
    test_cost, test_acc, eachAcc,preds, test_duration = evaluate(sess, features, support, y_test, test_mask, placeholders)
    return test_acc,eachAcc,preds

# if __name__ == '__main__':
# GCN_Train()