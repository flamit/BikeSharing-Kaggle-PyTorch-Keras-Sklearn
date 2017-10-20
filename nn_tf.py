import numpy as np
from data_utils import DataUtils as du
import tensorflow as tf
import matplotlib.pyplot as plt

#TOTAL_DATASET_SIZE = 10887
epochs = 10000
batch_size = 64
layer_dims = {"in": 13, "fc1": 10, "fc2": 10, "fc3": 10,"fc4": 10, "out": 1}
num_eval = 500

if __name__ == '__main__':
    _,_,_,X_train, Y_train, Y_train_log,X_val, Y_val,X_test,test_date_df = du.get_processed_df('data/train.csv', 'data/test.csv')

    train_data_size = X_train.shape[0]
    print(train_data_size)
    steps_in_epoch = train_data_size // batch_size

    #Dividing data to train and val datasets, conversion from DF to numpyarray
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Y_train_log = np.array(Y_train_log)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_test = np.array(X_test)

    # Defining input/output placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, layer_dims["in"]])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # weights and biases
    W = {
        "fc1": tf.get_variable("W1", shape=[layer_dims["in"], layer_dims["fc1"]],initializer=tf.contrib.layers.xavier_initializer()),
        "fc2": tf.get_variable("W2", shape=[layer_dims["fc1"], layer_dims["fc2"]], initializer=tf.contrib.layers.xavier_initializer()),
        "fc3": tf.get_variable("W3", shape=[layer_dims["fc2"], layer_dims["fc3"]], initializer=tf.contrib.layers.xavier_initializer()),
        "fc4": tf.get_variable("W4", shape=[layer_dims["fc3"], layer_dims["fc4"]], initializer=tf.contrib.layers.xavier_initializer()),
        "out": tf.get_variable("W5", shape=[layer_dims["fc4"], layer_dims["out"]], initializer=tf.contrib.layers.xavier_initializer()),
    }

    B = {
        "fc1": tf.get_variable("B1", shape=[layer_dims["fc1"]], initializer=tf.zeros_initializer()),
        "fc2": tf.get_variable("B2", shape=[layer_dims["fc2"]], initializer=tf.zeros_initializer()),
        "fc3": tf.get_variable("B3", shape=[layer_dims["fc3"]], initializer=tf.zeros_initializer()),
        "fc4": tf.get_variable("B4", shape=[layer_dims["fc4"]], initializer=tf.zeros_initializer()),
        "out": tf.get_variable("B5", shape=[layer_dims["out"]], initializer=tf.zeros_initializer()),
    }

    # Defining our model
    fc1 = tf.add(tf.matmul(X, W['fc1']), B['fc1'])
    fc1 = tf.nn.tanh(fc1)
    fc2 = tf.add(tf.matmul(fc1, W['fc2']), B['fc2'])
    fc2 = tf.nn.tanh(fc2)
    fc3 = tf.add(tf.matmul(fc2, W['fc3']), B['fc3'])
    fc3 = tf.nn.tanh(fc3)
    fc4 = tf.add(tf.matmul(fc3, W['fc4']), B['fc4'])
    fc4 = tf.nn.tanh(fc4)

    # Output
    output = tf.add(tf.matmul(fc4, W['out']), B['out'])
    output = tf.nn.relu(output)

    # Define loss and optimizer
    log_err = tf.log(output + 1) - tf.log(Y + 1)
    squared_le = tf.pow(log_err,2)
    mean_sle = tf.reduce_mean(squared_le)
    rmsle = tf.sqrt(mean_sle)

    #a = np.array([1,2,3,4,5,6,7,8,9,0])
    #b = np.array([1,2,3,4,5,5,4,3,2,1])
    #print(tf.log(a + 1) - tf.log(b + 1))

    loss_op = rmsle

    optimizer = tf.train.AdadeltaOptimizer()
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        best_val_loss = 1000
        best_train_loss = 1000
        val_err_his = np.zeros(epochs)
        train_err_his = np.zeros(epochs)
        for epoch in range(epochs):
            batch_ind = 0
            # Eval train error
            #print(X_train[:num_eval].shape,type(X_train))
            train_loss = sess.run(loss_op, feed_dict={X: X_train, Y: Y_train})
            train_err_his[epoch] = train_loss
            if train_loss < best_train_loss and epoch != 0:
                best_train_loss = train_loss
                acc_info = "train acc has improved!"
            else:
                acc_info = ""
            # Eval val error
            val_loss = sess.run(loss_op, feed_dict={X: X_val, Y: Y_val})
            val_err_his[epoch] = val_loss
            if val_loss < best_val_loss and epoch != 0:
                best_val_loss = val_loss
                acc_info += "val acc has improved! Model saved."
                saver.save(sess, "/tmp/model.ckpt")
            else:
                acc_info += ""
            print("Accuracy after", str(epoch), "epochs: Train set:", train_loss, ", Val set:", val_loss, acc_info)
            for step in range(steps_in_epoch):
                batch_x = X_train[batch_ind:batch_ind + batch_size]
                batch_y = Y_train[batch_ind:batch_ind + batch_size]
                # print(batch_x,batch_y)
                # Run optimization op (backprop)
                _, pred, loss_ = sess.run([train_op, output, loss_op],
                                          feed_dict={X: batch_x, Y: batch_y})
                #print(loss_)
                batch_ind += batch_size
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model loaded.")
        predictions = sess.run(output, feed_dict={X: X_test})

    test_date_df['count'] = np.array(predictions)
    test_date_df.to_csv("predictions.csv",index=False)

    plt.plot(val_err_his)
    plt.plot(train_err_his)
    plt.show()