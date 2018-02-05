import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import MyConfig as myCfg

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

learning_rate = 0.00001

INPUT_SIZE = 1024
NOISE_SIZE = 128
HIDDEN_SIZE = 256

total_epoch = 1000
total_size = 10000
batch_size = 200

# input placeholder
X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
# noise
Z = tf.placeholder(tf.float32, [None, NOISE_SIZE])

# generator's parameters
G_w1 = tf.Variable(tf.random_normal([NOISE_SIZE, HIDDEN_SIZE]))
G_b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))

G_w2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]))
G_b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))

G_w3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, INPUT_SIZE]))
G_b3 = tf.Variable(tf.random_normal([INPUT_SIZE]))


# discriminator's parameters
D_w1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE]))
D_b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))

D_w2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]))
D_b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))

# D_w3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE]))
# D_b3 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))

D_w3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, 1]))
D_b3 = tf.Variable(tf.random_normal([1]))


# GENERATOR - noise_z: 입력
def generator(noize_z):
    hidden = tf.nn.relu(tf.matmul(noize_z, G_w1) + G_b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden, G_w2) + G_b2)
    output = tf.nn.relu(tf.matmul(hidden2, G_w3) + G_b3)

    return output


# DISCRIMINATOR - original: 입력 (만들고자 하는 데이터)
def discriminator(original):
    hidden = tf.nn.tanh(tf.matmul(original, D_w1) + D_b1)
    hidden2 = tf.nn.tanh(tf.matmul(hidden, D_w2) + D_b2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, D_w3) + D_b3)

    return output

# noise 생성
def generate_noise(batch_size, noise_size):
    return np.random.normal(size=(batch_size, noise_size))


G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))
#
# loss_G = tf.reduce_mean(tf.log(D_gene))
loss_D = tf.reduce_mean(tf.log(tf.clip_by_value(D_real, 1e-10, 1.0)) + tf.log(tf.clip_by_value(1-D_gene, 1e-10, 1.0)))

loss_G = tf.reduce_mean(tf.log(tf.clip_by_value(D_gene, 1e-10, 1.0)))

# D_var_list = [D_w1, D_b1, D_w2, D_b2, D_w3, D_b3, D_w4, D_b4]
D_var_list = [D_w1, D_b1, D_w2, D_b2, D_w3, D_b3]
G_var_list = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3]

train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_G, var_list=G_var_list)


#### training

input_queue = tf.train.string_input_producer([myCfg.route], shuffle=True, name='input_queue')
reader = tf.TextLineReader()
key, value = reader.read(input_queue)

record_defaults = [[0.] for x in range(1024)]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch = tf.train.batch([xy], batch_size=batch_size)
# print(train_x_batch)
# print(train_y_batch)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for epoch in range(total_epoch):
    loss_val_D = 0
    loss_val_G = 0
    for loop in range(int(total_size/batch_size)):
        x_batch = sess.run(train_x_batch)
        nomalized_x = MinMaxScaler(x_batch)
        # print(x_batch)
        # print(nomalized_x)
        noise = generate_noise(batch_size=batch_size, noise_size=NOISE_SIZE)

        _, lv_D = sess.run([train_D, loss_D], feed_dict={X: nomalized_x, Z: noise})
        _, lv_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

        loss_val_D += lv_D/int(total_size/batch_size)
        loss_val_G += lv_G/int(total_size/batch_size)


    if epoch == 0 or epoch % 50 == 0:
        generated_size = 5
        noise = generate_noise(batch_size=generated_size, noise_size=NOISE_SIZE)
        generated = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(generated_size, 1, figsize=(10, generated_size))

        for i in range(generated_size):
            # ax[i].set_axis_off()
            ax[i].plot(np.reshape(generated[i], (INPUT_SIZE, 1)))
            ax[i].grid()
            # ax[i].imshow(np.reshape(generated[i], (1, INPUT_SIZE)))

        plt.savefig('generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

        print('Epoch:', '%04d' % epoch,
              'D loss: {:.4}'.format(loss_val_D),
              'G loss: {:.4}'.format(loss_val_G))


coord.request_stop()
coord.join(threads)