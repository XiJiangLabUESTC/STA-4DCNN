from __future__ import division
from scipy.io import loadmat
import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
import scipy.stats as stats              
from scipy.stats import pearsonr
import scipy.io as scio
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_data():
    temp1 = []
    for i in tqdm(range(train_num)):
        train_path = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion/%s/input_data.mat' % (i + 1)
        train_data = loadmat(train_path)
        train_data = train_data['input_data']
        temp1.append(train_data)
    x = np.array(temp1)
    x = x.reshape(-1, 48, 56, 48, 88)  # x,y,z,t
    x = x.astype(np.float32)

    temp1 = []
    for i in tqdm(range(test_num)):
        train_path = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion/%s/input_data.mat' % (i + train_num + 1)
        train_data = loadmat(train_path)
        train_data = train_data['input_data']
        temp1.append(train_data)
    test_x = np.array(temp1)
    test_x = test_x.reshape(-1, 48, 56, 48, 88)  # x,y,z,t
    test_x = test_x.astype(np.float32)

    temp2 = []
    for i in range(train_num):
        path2 = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion_label/time/%s/time.mat' % (i + 1)
        y_t = loadmat(path2)
        y_t = y_t['time']
        temp2.append(y_t)
    y_t = np.array(temp2)
    y_t = y_t.astype(np.float32)

    temp2 = []
    for i in range(test_num):
        path2 = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion_label/time/%s/time.mat' % (i + train_num + 1)
        test_y_t = loadmat(path2)
        test_y_t = test_y_t['time']
        temp2.append(test_y_t)
    test_y_t = np.array(temp2)
    test_y_t = test_y_t.astype(np.float32)

    temp2 = []
    for i in range(train_num):
        path2 = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch150/%s/space.mat' % (i + 1)
        spatial_p = loadmat(path2)
        spatial_p = spatial_p['space']
        temp2.append(spatial_p)
    spatial_p = np.array(temp2)
    spatial_p = spatial_p.reshape(-1, 48, 56, 48)
    spatial_p = spatial_p.astype(np.float32)

    temp2 = []
    for i in range(test_num):
        path2 = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch150/%s/space.mat' % (i + train_num + 1)
        test_spatial_p = loadmat(path2)
        test_spatial_p = test_spatial_p['space']
        temp2.append(test_spatial_p)
    test_spatial_p = np.array(temp2)
    test_spatial_p = test_spatial_p.reshape(-1, 48, 56, 48)
    test_spatial_p = test_spatial_p.astype(np.float32)

    return x, test_x, y_t, test_y_t, spatial_p, test_spatial_p


def Attentionnet(x,spatial):
    with tf.variable_scope("attention1") as scope:
        #key and value
        x1=tf.transpose(x, [0,2,3,4,1])
        filter_sets1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 48, 12], stddev=0.1))
        c1 = tf.nn.conv3d(x1, filter_sets1, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn1 = tf.keras.layers.BatchNormalization()
        c1_bn = bn1(c1, training=True)
        m1 = tf.nn.relu(c1_bn)
        x2=tf.transpose(m1, [0,4,2,3,1])
        filter_sets2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 56, 12], stddev=0.1))
        c2 = tf.nn.conv3d(x2, filter_sets2, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn2 = tf.keras.layers.BatchNormalization()
        c2_bn = bn2(c2, training=True)
        m2 = tf.nn.relu(c2_bn)
        x3=tf.transpose(m2, [0,1,4,3,2])
        filter_sets3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 48, 12], stddev=0.1))
        c3 = tf.nn.conv3d(x3, filter_sets3, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn3 = tf.keras.layers.BatchNormalization()
        c3_bn = bn3(c3, training=True)
        m3 = tf.nn.relu(c3_bn)
        temp_key1 = tf.reshape(tf.transpose(m3, [0,1,2,4,3]),[batch_size,1728,88])
        temp_value1 = tf.reshape(tf.transpose(m3, [0,1,2,4,3]),[batch_size,1728,88])

        #query
        temp_x1 = tf.reshape(x,[batch_size,129024,88])
        temp_query1 = tf.tile(tf.reshape(spatial,[batch_size,129024,1]),[1,1,88])
        query1 = tf.multiply(temp_x1,temp_query1, name='query1_result') #b*129024*88
        #query1 = tf.tile(tf.reshape(spatial,[batch_size,48,56,48,1]),[1,1,1,1,88])
        query_temp = tf.reshape(query1,[batch_size,48,56,48,88])
        x4=tf.transpose(query_temp, [0,2,3,4,1])
        filter_sets4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 48, 12], stddev=0.1))
        c4 = tf.nn.conv3d(x4, filter_sets4, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn4 = tf.keras.layers.BatchNormalization()
        c4_bn = bn4(c4, training=True)
        m4 = tf.nn.relu(c4_bn)
        x5=tf.transpose(m4, [0,4,2,3,1])
        filter_sets5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 56, 12], stddev=0.1))
        c5 = tf.nn.conv3d(x5, filter_sets5, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn5 = tf.keras.layers.BatchNormalization()
        c5_bn = bn5(c5, training=True)
        m5 = tf.nn.relu(c5_bn)
        x6=tf.transpose(m5, [0,1,4,3,2])
        filter_sets6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 48, 12], stddev=0.1))
        c6 = tf.nn.conv3d(x6, filter_sets6, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn6 = tf.keras.layers.BatchNormalization()
        c6_bn = bn6(c6, training=True)
        m6 = tf.nn.relu(c6_bn)
        temp_query1 = tf.reshape(tf.transpose(m6, [0,1,2,4,3]),[batch_size,1728,88])

        #calculate attention
        temp_attention1 = tf.matmul(temp_query1,tf.transpose(temp_key1, [0,2,1])) #b*query*key
        temp_attention2 = tf.nn.softmax(temp_attention1,2)
        attention_output1 = tf.transpose(tf.matmul(temp_attention2,temp_value1), [0,2,1]) #b*88*query
        attention_output2 = tf.reduce_mean(attention_output1,2) #b*88
        #bias_init7 = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32))
        #filter_sets7 = tf.Variable(tf.truncated_normal(shape=[3,1000,1], stddev=0.1))
        #c7 = tf.nn.conv1d(attention_output1, filter_sets7, 1, padding='SAME')
        #c7_temp = tf.nn.bias_add(c7, bias_init7)
        #attention1_output = tf.nn.relu(c7_temp)
        
    result_time = tf.reshape(attention_output2,[batch_size,88], name='result_t')
    return result_time

# input
batch_size = 4
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 48, 56, 48, 88], name='input_x')
spatial = tf.placeholder(dtype=tf.float32, shape=[batch_size, 48, 56, 48], name='spatial_pattern')
y_t = tf.placeholder(dtype=tf.float32, shape=[batch_size, 88], name='label_t')
train_num = 160
test_num = 40
epoch_num = 20
lr = 0.0005

def pearson_corr(y_t, predict_t):
    n = predict_t.shape[0].value
    n = tf.cast(n,dtype=tf.float32)
    sum_y = tf.reduce_sum(y_t)
    sum_p = tf.reduce_sum(predict_t)
    sum_y_sq = tf.reduce_sum(tf.square(y_t))
    sum_p_sq = tf.reduce_sum(tf.square(predict_t))
    sum_mul= tf.reduce_sum(tf.multiply(y_t, predict_t))
    unmerator = tf.subtract(tf.multiply(n,sum_mul),tf.multiply(sum_y,sum_p))
    denominator1 = tf.subtract(tf.multiply(n,sum_y_sq),tf.square(sum_y))
    denominator2 = tf.subtract(tf.multiply(n,sum_p_sq),tf.square(sum_p))
    denominator = tf.multiply(tf.sqrt(denominator1),tf.sqrt(denominator2))
    corr = tf.divide(unmerator,denominator)
    return tf.cond(tf.is_nan(corr), lambda: 0., lambda: corr)

predict_t = Attentionnet(x,spatial)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(y_t[i,:],[88]), tf.reshape(predict_t[i,:],[88]))
loss_array = tf.stack(loss_temp, axis=0)
loss = tf.reduce_mean(loss_array)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# save
saver = tf.train.Saver(max_to_keep=100)

# begin train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
x_r, test_x_r, y_t_r, test_y_t_r, spatial_r, test_spatial_r = load_data()
for epoch in tqdm(range(epoch_num)):
    path = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result_mat/epoch%s' % epoch
    os.mkdir(path)
    epoch_loss = 0
    val_loss = 0
    for i in range(int((train_num - 1) / batch_size) + 1):
        min_end = min(batch_size * (i + 1), train_num)
        epoch_x, epoch_y_t, epoch_spatial = x_r[i * batch_size:min_end, :, :, :, :], y_t_r[i * batch_size:min_end,:].reshape(batch_size,88), spatial_r[i * batch_size:min_end,:, :, :].reshape(batch_size,48, 56, 48)
        sess.run(optimizer, feed_dict={x: epoch_x, y_t: epoch_y_t, spatial: epoch_spatial})
    saver.save(sess, r'/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/Xinet.ckpt-' + str(epoch))
    
    #cal_loss
    for i in range(int((train_num - 1) / batch_size) + 1):
        min_end = min(batch_size * (i + 1), train_num)
        epoch_x, epoch_y_t, epoch_spatial = x_r[i * batch_size:min_end, :, :, :, :], y_t_r[i * batch_size:min_end,:].reshape(batch_size,88), spatial_r[i * batch_size:min_end,:, :, :].reshape(batch_size,48, 56, 48)
        loss_total, test_t = sess.run([loss, predict_t],feed_dict={x: epoch_x, y_t: epoch_y_t, spatial: epoch_spatial})
        epoch_loss += loss_total / (int((train_num - 1) / batch_size) + 1)
        path = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result_mat/epoch%s/%s' %(epoch,i+1)
        os.mkdir(path)
        save2 = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result_mat/epoch%s/%s/time.mat' %(epoch,i+1)
        scio.savemat(save2, {'time':test_t})
    print('Epoch: %03d/%03d time_loss: %.9f' % (epoch, epoch_num, epoch_loss))
    f = open('/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result', 'a')
    f.write('\n' + str(epoch_loss))
    f.close()

    for i in range(int((test_num - 1) / batch_size) + 1):
        min_end = min(batch_size * (i + 1), test_num)
        val_x, val_y_t, val_spatial = test_x_r[i * batch_size:min_end, :, :, :, :], test_y_t_r[i * batch_size:min_end,:].reshape(batch_size,88), test_spatial_r[i * batch_size:min_end,:, :, :].reshape(batch_size,48, 56, 48)
        loss_total, test_t = sess.run([loss, predict_t],feed_dict={x: val_x, y_t: val_y_t, spatial: val_spatial})
        val_loss += loss_total / (int((test_num - 1) / batch_size) + 1)
        path = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result_mat/epoch%s/%s' %(epoch,(int((train_num - 1) / batch_size) + 1)+i+1)
        os.mkdir(path)
        save2 = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result_mat/epoch%s/%s/time.mat' %(epoch,(int((train_num - 1) / batch_size) + 1)+i+1)
        scio.savemat(save2, {'time':test_t})
    print('Epoch: %03d/%03d time_val: %.9f' % (epoch, epoch_num, val_loss))
    f = open('/media/D/alex/NEW_GA4DCNN/medical_image_analysis/temporal/proposed/result', 'a')
    f.write('\n' + str(val_loss))
    f.close()

