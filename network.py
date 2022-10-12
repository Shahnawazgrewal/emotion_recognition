import tensorflow as tf

# Network Parameters
n_hidden_1 = 200  # 1st layer number of neurons
n_hidden_2 = 200  # 2nd layer number of neurons
n_input    = 512  # data input
n_classes  = 7  # total classes





with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None, n_input), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')

global_step = tf.Variable(0, trainable=False, name='global_step')


def build_network(input_images, labels, weigths, biases, num_classes):
    logits = multilayer_perceptron(input_images, weigths, biases)

    with tf.variable_scope('loss') as scope:
        with tf.name_scope('soft_loss'):
            softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        scope.reuse_variables()
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('TotalLoss', softmax)

    return logits, softmax, accuracy  # returns total loss


def multilayer_perceptron(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully cn_classesonnected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
