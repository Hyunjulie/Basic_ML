import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(999)
tf.reset_default_graph()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Setting hyperparameters
learning_rate = 1e-3
training_epochs = 10
batch_size = 100
p_keep_conv_value = 0.8
keep_prob = 0.5

class CNN: 
  
  def __init__ (self, sess, name):
    self.sess = sess
    self.name = name
    self.build_net()
    
  def build_net(self):
    with tf.variable_scope(self.name):
      self.training = tf.placeholder(tf.bool)

      # placeholders for input 
      self.X = tf.placeholder(tf.float32, [None, 784])
      X_img = tf.reshape(self.X, [-1, 28, 28, 1])
      self.Y = tf.placeholder(tf.float32, [None, 10])

      # Layer 1: Convolution, pooling
      conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding="SAME", strides=2)
      dropout1 = tf.layers.dropout(inputs=pool1, rate = 0.5, training=self.training)

      # Layer 2: Convolution, pooling
      conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], padding="SAME", strides=2)
      dropout2 = tf.layers.dropout(inputs=pool2, rate = 0.3, training=self.training)

      # Layer 3: Convolution, pooling
      conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
      pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding="SAME", strides=2)
      dropout3 = tf.layers.dropout(inputs=pool3, rate = 0.3, training=self.training)

      # Layer 4: Dense Layer with Relu 
      flat = tf.reshape(dropout3, [-1, 128*4*4])
      dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
      dropout4 = tf.layers.dropout(inputs=dense4, rate = 0.5, training=self.training)

      # Layer 5: FC 625 inputs --> 10 outputs 
      self.logits = tf.layers.dense(inputs=dropout4, units=10)
    
    # Cost/Loss functions, Optimizer 
    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    
    correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  def predict(self, x_test, training=False):
    return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})
  
  def get_accuracy(self, x_test, y_test, training=False):
    return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})
  
  def train(self, x_data, y_data, training=True):
    return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y:y_data, self.training: training})


# Initialization 
sess = tf.Session()
model = CNN(sess, "model")

sess.run(tf.global_variables_initializer())

print("Learning started")

#Training 'model'
for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = int(mnist.train.num_examples / batch_size)
  
  for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    c, _ = model.train(batch_xs, batch_ys)
    avg_cost += c / total_batch
  
  print("Epoch: ", '%04d' %(epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))
  
print('Learning Finished')

# Testing model and check accuracy 
print('Accuracy: ', model.get_accuracy(mnist.test.images, mnist.test.labels))


'''
With 2 epochs 

Learning started
Epoch:  0001 cost:  0.328968363
Epoch:  0002 cost:  0.105396725
Learning Finished
Accuracy:  0.9873

'''