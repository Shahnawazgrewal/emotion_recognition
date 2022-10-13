import os
import argparse
import sys
import tensorflow as tf
import dataset_utils
import network
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])


def main(_):
    # read train and test embedding files
    train_file = FLAGS.train_file
    test_file = FLAGS.test_file
    train_set, train_label = dataset_utils.read_file(train_file)
    test_set, test_label = dataset_utils.read_file(test_file)
    # dataset statistics
    print("Train file length", len(train_set))
    print("Train file label length", len(train_label))
    print("Test file length", len(test_set))
    print("Test label length", len(test_label))

    clabels = train_label + test_label
    ilabels = np.unique(clabels, return_inverse=True)[1]

    tmp = list(split(ilabels, 2))
    train_label = tmp[0]
    test_label = tmp[1]

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([FLAGS.input_emb_size, FLAGS.n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([FLAGS.n_hidden_1, FLAGS.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden_2, FLAGS.n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([FLAGS.n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([FLAGS.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
    }

    mean_data_img_train = np.mean(train_set, axis=0)

    with tf.name_scope('input'):
        input_images = tf.placeholder(tf.float32, shape=(None, FLAGS.input_emb_size), name='input_images')
        labels = tf.placeholder(tf.int64, shape=(None), name='labels')

    features, total_loss, accuracy = network.build_network(input_images, labels, weights,
                                                           biases, FLAGS.n_classes)

    FLAGS.batch_size = 32

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(0.001)  # learning rate.
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_train_samples = len(train_set)
    num_test_samples = len(test_set)
    num_of_batches = num_train_samples // FLAGS.batch_size
    num_of_batches_test = num_test_samples // FLAGS.batch_size

    for epoch in range(FLAGS.epochs):
        test_acc = 0.
        train_acc = 0.
        train_loss = 0.
        test_loss = 0.
        for idx in range(num_of_batches):
            batch_images, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, train_set)
            _, summary_str, train_batch_acc, train_batch_loss = sess.run(
                [train_op, summary_op, accuracy, total_loss],
                feed_dict={
                    input_images: batch_images - mean_data_img_train,
                    labels: batch_labels,
                })

            train_acc += train_batch_acc
            train_loss += train_batch_loss

        train_acc /= num_of_batches
        train_acc = train_acc * 100

        for s_batch in range(num_of_batches_test):
            batch_images_test, batch_labels_test = get_batch(s_batch, FLAGS.batch_size, test_label, test_set)
            _, summary_str, test_batch_acc, test_loss_batch = sess.run(
                [train_op, summary_op, accuracy, total_loss],
                feed_dict={
                    input_images: batch_images_test,
                    labels: batch_labels_test,
                })
            test_acc += test_batch_acc
            test_loss += test_loss_batch

        test_acc /= num_of_batches_test
        test_acc = test_acc * 100

        print(("Epoch: {}, Train_Acc:{:.4f}, Train_Loss:{:.4f}, Test_Acc:{:.4f}, Test_loss:{:.4f}".
               format(epoch, train_acc, train_loss, test_acc, test_loss)))


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--train_file', type=str, default='voxFeats/train.pkl', help='Path to the trail feature file.')
    parser.add_argument('--test_file', type=str, default='voxFeats/test.pkl', help='Path to the test feature file.')
    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--input_emb_size', type=int, default=512, help='Input embedding size.')
    parser.add_argument('--n_hidden_1', type=int, default=200, help='Batch size for training.')
    parser.add_argument('--n_hidden_2', type=int, default=200, help='Batch size for training.')
    parser.add_argument('--n_classes', type=int, default=7, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Batch size for training.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
