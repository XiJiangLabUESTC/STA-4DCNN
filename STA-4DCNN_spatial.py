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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_data():
    temp1 = []
    for i in tqdm(range(train_num)):
        train_path = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion/%s/input_data.mat' % (i + 1)
        train_data = loadmat(train_path)
        train_data = train_data['input_data']
        temp1.append(train_data)
    x = np.array(temp1)
    x = x.reshape(-1, 48, 56, 48, 88, 1)  # x,y,z,t
    x = x.astype(np.float32)

    temp1 = []
    for i in tqdm(range(test_num)):
        train_path = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion/%s/input_data.mat' % (i + train_num + 1)
        train_data = loadmat(train_path)
        train_data = train_data['input_data']
        temp1.append(train_data)
    test_x = np.array(temp1)
    test_x = test_x.reshape(-1, 48, 56, 48, 88, 1)  # x,y,z,t
    test_x = test_x.astype(np.float32)

    temp1 = []
    for i in range(train_num):
        path1 = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion_label/space/%s/space.mat' % (i + 1)
        y_s = loadmat(path1)
        y_s = y_s['space']
        temp1.append(y_s)
    y_s = np.array(temp1)
    y_s = y_s.astype(np.float32)

    temp1 = []
    for i in range(test_num):
        path1 = '/media/D/alex/NEW_GA4DCNN/EMOTION/emotion_label/space/%s/space.mat' % (i + train_num + 1)
        test_y_s = loadmat(path1)
        test_y_s = test_y_s['space']
        temp1.append(test_y_s)
    test_y_s = np.array(temp1)
    test_y_s = test_y_s.astype(np.float32)

    return x, y_s, test_x, test_y_s


def Maxpooling4d(
        input,
        ksize=[2, 2, 2, 2],
        strides=[2, 2, 2, 2],
        padding='VALID',
        data_format='NDHWC'
        ):

    (b, l_i, d_i, h_i, w_i, c_i) = input.shape
    (b, l_i, d_i, h_i, w_i, c_i) = (int(b), int(l_i), int(d_i), int(h_i), int(w_i), int(c_i))

    frame_results = [None]*w_i

    for i in range(w_i):
        frame_Maxpooling4d = tf.nn.max_pool3d(tf.reshape(input[:, :, :, :, i, :], (b, l_i, d_i, h_i, c_i)),
                                            [1, ksize[0], ksize[1], ksize[2], 1],
                                            strides=[1, strides[0], strides[1], strides[2], 1], padding='VALID',
                                            data_format="NDHWC")
        frame_results[i] = frame_Maxpooling4d

    result3d = tf.stack(frame_results, axis=4)

    output_t = int(w_i / 2)
    result4d = [None]*output_t
    for i in range(output_t):
        temp1 = tf.reshape(result3d[:, :, :, :, 2*i, :],(int(b), int(l_i/2), int(d_i/2), int(h_i/2), 1, int(c_i)))
        temp2 = tf.reshape(result3d[:, :, :, :, 2*i+1, :],(int(b), int(l_i/2), int(d_i/2), int(h_i/2), 1, int(c_i)))
        temp = tf.concat([temp1, temp2], axis=4)
        result4d[i] = tf.reduce_max(temp, axis=4, keepdims=False)
    output = tf.stack(result4d, axis=4)
    return output

def Avgpooling4d(
        input,
        ksize=[2, 2, 2],
        strides=[2, 2, 2],
        padding='VALID',
        data_format='NDHWC'
        ):

    (b, l_i, d_i, h_i, w_i, c_i) = input.shape
    (b, l_i, d_i, h_i, w_i, c_i) = (int(b), int(l_i), int(d_i), int(h_i), int(w_i), int(c_i))
    t=int(w_i/2)

    frame_results = [None]*t #t take the 2*i

    for i in range(t):
        frame_Maxpooling4d = tf.nn.avg_pool3d(tf.reshape(input[:, :, :, :, 2*i+1, :], (b, l_i, d_i, h_i, c_i)),
                                            [1, ksize[0], ksize[1], ksize[2], 1],
                                            strides=[1, strides[0], strides[1], strides[2], 1], padding='VALID',
                                            data_format="NDHWC")
        frame_results[i] = frame_Maxpooling4d

    output = tf.stack(frame_results, axis=4)
    #Avgpooling4d(x,[2,2,2],[2,2,2])
    return output

def conv4d(
        input,
        filter_sets,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name=None):
    (l_k, d_k, h_k, w_k, in_c, out_c) = filter_sets.shape
    (l_k, d_k, h_k, w_k, in_c, out_c) = (int(l_k), int(d_k), int(h_k), int(w_k), int(in_c), int(out_c))
    # output size for 'valid' convolution
    if padding == 'VALID':
        # (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.get_shape().as_list())
        (b, l_i, d_i, h_i, w_i, c_i) = input.shape
        (b, l_i, d_i, h_i, w_i, c_i) = (int(b), int(l_i), int(d_i), int(h_i), int(w_i), int(c_i))
        (l_o, d_o, h_o, w_o) = (
            (l_i - l_k) / strides[0] + 1,
            (d_i - d_k) / strides[1] + 1,
            (h_i - h_k) / strides[2] + 1,
            (w_i - w_k) / strides[3] + 1
        )
        (l_o, d_o, h_o, w_o) = (int(l_o), int(d_o), int(h_o), int(w_o))
        compen = int(l_k / 2)
    else:
        padding_l = int((l_k - 1) / 2)
        padding_d = int((d_k - 1) / 2)
        padding_h = int((h_k - 1) / 2)
        padding_w = int((w_k - 1) / 2)
        input = tf.pad(input, tf.constant([[0, 0], [padding_l, padding_l], [padding_d, padding_d], [padding_h, padding_h],[padding_w, padding_w], [0, 0]]), "CONSTANT")
        (b, l_i, d_i, h_i, w_i, c_i) = tuple(input.get_shape().as_list())
        (l_o, d_o, h_o, w_o) = (
            (l_i - l_k) / strides[0] + 1,
            (d_i - d_k) / strides[1] + 1,
            (h_i - h_k) / strides[2] + 1,
            (w_i - w_k) / strides[3] + 1
        )
        (l_o, d_o, h_o, w_o) = (int(l_o), int(d_o), int(h_o), int(w_o))
        compen = int(l_k / 2)

    # output tensors for each 3D frame
    frame_results = [None] * l_o

    # convolve each kernel frame i with each input frame j
    for j in np.arange(compen, l_i, strides[0]):  # range(l_i):

        # reuse variables of previous 3D convolutions for the same kernel
        # frame (or if the user indicated to have all variables reused)
        # reuse_kernel = reuse
        out_frame = int((j - compen) / strides[0])
        if out_frame >= l_o:
            continue
        tmp = None
        for i in range(l_k):
            if j - int(l_k / 2) + i < 0 or j - int(l_k / 2) + i >= l_i:
                continue
            frame_conv3d = tf.nn.conv3d(tf.reshape(input[:, j - int(l_k / 2) + i, :, :, :, :], (b, d_i, h_i, w_i, c_i)),filter_sets[i, :, :, :, :, :],
                                        strides=[1, strides[1], strides[2], strides[3], 1], padding='VALID',data_format="NDHWC", name=name + '_3dchan%d' % i)
            if tmp is None:
                tmp = frame_conv3d
            else:
                tmp += frame_conv3d
        frame_results[out_frame] = tmp

    output = tf.stack(frame_results, axis=1)
    return output

def Conv3d(x, k_size, s_size, c_in, c_out, pad_type, filter_name):
    filter_sets = tf.get_variable(filter_name, shape=[k_size, k_size, k_size, c_in, c_out], initializer=tf.keras.initializers.he_normal())
    c = tf.nn.conv3d(x, filter_sets, strides=[1, s_size, s_size, s_size, 1], padding=pad_type)
    #bn = tf.keras.layers.BatchNormalization()
    #c_temp = bn(c, training=True)
    #output = tf.nn.relu(c_temp)
    return c

def attention_conv(input, name_string):
    (b, l_i, d_i, h_i, w_i, c_i) = input.shape
    (b, l_i, d_i, h_i, w_i, c_i) = (int(b), int(l_i), int(d_i), int(h_i), int(w_i), int(c_i))
    attention_result = [None]*c_i
    for i in range(c_i):
        weight_q = name_string+'q%s' %i
        weight_k = name_string+'k%s' %i
        weight_v = name_string+'v%s' %i
        sub_x = tf.reshape(input[:, :, :, :, :, i], (b, l_i, d_i, h_i, w_i))
        temp_q = tf.reshape(Conv3d(sub_x, 3, 1, w_i, w_i, 'SAME', weight_q), [l_i*d_i*h_i, w_i])
        temp_k = tf.reshape(Conv3d(sub_x, 3, 1, w_i, w_i, 'SAME', weight_k), [l_i*d_i*h_i, w_i])
        temp_v = tf.reshape(Conv3d(sub_x, 3, 1, w_i, w_i, 'SAME', weight_v), [l_i*d_i*h_i, w_i])
        temp_A = tf.matmul(temp_q, tf.transpose(temp_k, [1, 0]))
        A = tf.nn.softmax(temp_A, 1)
        temp_output = tf.reshape(tf.matmul(A, temp_v), [b, l_i, d_i, h_i, w_i])
        attention_result[i] = temp_output
    bn = tf.keras.layers.BatchNormalization()
    output = tf.reshape(tf.nn.relu(bn(tf.stack(attention_result, axis=1), training=True)), [b, l_i, d_i, h_i, w_i, c_i])
    return output


def deconv(input, padding_size=[1, 1, 1, 1]):
    (b, l_i, d_i, h_i, w_i, c_i) = input.shape
    (b, l_i, d_i, h_i, w_i, c_i) = (int(b), int(l_i), int(d_i), int(h_i), int(w_i), int(c_i))
    indices = []
    for i in range(l_i):
        for j in range(d_i):
            for k in range(h_i):
                for l in range(w_i):
                    for m in range(c_i):
                        indices.append([0, 2 * i + padding_size[0] , 2 * j + padding_size[1], 2 * k + padding_size[2], 2 * l + padding_size[3], m])
    values = tf.reshape(input, [-1])
    temp = tf.SparseTensor(indices, values, dense_shape=[b, 2 * l_i + padding_size[0], 2 * d_i + padding_size[1], 2 * h_i + padding_size[2], 2 * w_i + padding_size[3], c_i])
    result = tf.sparse_tensor_to_dense(temp)
    return result

def Xinet(x):
    base_num=1
    with tf.variable_scope("conv1") as scope:
        filter_sets = tf.get_variable('W1', shape=[3, 3, 3, 3, 1, base_num], initializer=tf.keras.initializers.he_normal())
        c1 = conv4d(input=x, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv1')
        bn1 = tf.keras.layers.BatchNormalization()
        c1_bn = bn1(c1, training=True)
        m1 = tf.nn.relu(c1_bn, name='conv1_result')

    with tf.variable_scope("conv2") as scope:
        filter_sets = tf.get_variable('W2', shape=[3, 3, 3, 3, base_num, base_num], initializer=tf.keras.initializers.he_normal())
        c2 = conv4d(input=m1, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv2')
        bn2 = tf.keras.layers.BatchNormalization()
        c2_bn = bn2(c2, training=True)
        m2 = tf.nn.relu(c2_bn, name='conv2_result')

    pooling1 = Maxpooling4d(m2, [2, 2, 2, 2], [2, 2, 2, 2])

    with tf.variable_scope("conv3") as scope:
        filter_sets = tf.get_variable('W3', shape=[3, 3, 3, 3, base_num, 2*base_num], initializer=tf.keras.initializers.he_normal())
        c3 = conv4d(input=pooling1, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv3')
        bn3 = tf.keras.layers.BatchNormalization()
        c3_bn = bn3(c3, training=True)
        m3 = tf.nn.relu(c3_bn, name='conv3_result')

    with tf.variable_scope("conv4") as scope:
        filter_sets = tf.get_variable('W4', shape=[3, 3, 3, 3, 2*base_num, 2*base_num], initializer=tf.keras.initializers.he_normal())
        c4 = conv4d(input=m3, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv4')
        bn4 = tf.keras.layers.BatchNormalization()
        c4_bn = bn4(c4, training=True)
        m4 = tf.nn.relu(c4_bn, name='conv4_result')

    pooling2 = Maxpooling4d(m4, [2, 2, 2, 2], [2, 2, 2, 2])

    with tf.variable_scope("conv5") as scope:
        filter_sets = tf.get_variable('W5', shape=[3, 3, 3, 3, 2*base_num, 4*base_num], initializer=tf.keras.initializers.he_normal())
        c5 = conv4d(input=pooling2, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv5')
        bn5 = tf.keras.layers.BatchNormalization()
        c5_bn = bn5(c5, training=True)
        m5 = tf.nn.relu(c5_bn, name='conv5_result')

    with tf.variable_scope("conv6") as scope:
        filter_sets = tf.get_variable('W6', shape=[3, 3, 3, 3, 4*base_num, 4*base_num], initializer=tf.keras.initializers.he_normal())
        c6 = conv4d(input=m5, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv6')
        bn6 = tf.keras.layers.BatchNormalization()
        c6_bn = bn6(c6, training=True)
        m6 = tf.nn.relu(c6_bn, name='conv6_result')

    pooling3 = Maxpooling4d(m6, [2, 2, 2, 2], [2, 2, 2, 2])

    with tf.variable_scope("conv7") as scope:
        filter_sets = tf.get_variable('W7', shape=[3, 3, 3, 3, 4*base_num, 8*base_num], initializer=tf.keras.initializers.he_normal())
        c7 = conv4d(input=pooling3, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv7')
        bn7 = tf.keras.layers.BatchNormalization()
        c7_bn = bn7(c7, training=True)
        m7 = tf.nn.relu(c7_bn, name='conv7_result')
    
    with tf.variable_scope("conv8") as scope:
        filter_sets = tf.get_variable('W8', shape=[3, 3, 3, 3, 8*base_num, 8*base_num], initializer=tf.keras.initializers.he_normal())
        c8 = conv4d(input=m7, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv8')
        bn8 = tf.keras.layers.BatchNormalization()
        c8_bn = bn8(c8, training=True)
        m8 = tf.nn.relu(c8_bn, name='conv8_result')

    with tf.variable_scope("deconv1") as scope:
        upsampling1=deconv(m8,[0,0,0,0])
        filter_sets = tf.get_variable('WD1', shape=[3, 3, 3, 3, 8*base_num, 8*base_num], initializer=tf.keras.initializers.he_normal())
        d1 = conv4d(input=upsampling1, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='deconv1')
        dbn1 = tf.keras.layers.BatchNormalization()
        d1_bn = dbn1(d1, training=True)
        dm1 = tf.nn.relu(d1_bn, name='deconv1_result')

    copy1 = tf.concat([attention_conv(m6, 'weight_attention1'), dm1], 5)

    with tf.variable_scope("conv9") as scope:
        filter_sets = tf.get_variable('W9', shape=[3, 3, 3, 3, 12*base_num, 4*base_num], initializer=tf.keras.initializers.he_normal())
        c9 = conv4d(input=copy1, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv9')
        bn9 = tf.keras.layers.BatchNormalization()
        c9_bn = bn9(c9, training=True)
        m9 = tf.nn.relu(c9_bn, name='conv9_result')

    with tf.variable_scope("conv10") as scope:
        filter_sets = tf.get_variable('W10', shape=[3, 3, 3, 3, 4*base_num, 4*base_num], initializer=tf.keras.initializers.he_normal())
        c10 = conv4d(input=m9, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv10')
        bn10 = tf.keras.layers.BatchNormalization()
        c10_bn = bn10(c10, training=True)
        m10 = tf.nn.relu(c10_bn, name='conv10_result')

    with tf.variable_scope("deconv2") as scope:
        upsampling2=deconv(m10,[0,0,0,0])
        filter_sets = tf.get_variable('WD2', shape=[3, 3, 3, 3, 4*base_num, 4*base_num], initializer=tf.keras.initializers.he_normal())
        d2 = conv4d(input=upsampling2, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='deconv2')
        dbn2 = tf.keras.layers.BatchNormalization()
        d2_bn = dbn2(d2, training=True)
        dm2 = tf.nn.relu(d2_bn, name='deconv2_result')

    copy2 = tf.concat([attention_conv(m4, 'weight_attention2'), dm2], 5)

    with tf.variable_scope("conv11") as scope:
        filter_sets = tf.get_variable('W11', shape=[3, 3, 3, 3, 6*base_num, 2*base_num], initializer=tf.keras.initializers.he_normal())
        c11 = conv4d(input=copy2, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv11')
        bn11 = tf.keras.layers.BatchNormalization()
        c11_bn = bn11(c11, training=True)
        m11 = tf.nn.relu(c11_bn, name='conv11_result')

    with tf.variable_scope("conv12") as scope:
        filter_sets = tf.get_variable('W12', shape=[3, 3, 3, 3, 2*base_num, 2*base_num], initializer=tf.keras.initializers.he_normal())
        c12 = conv4d(input=m11, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv12')
        bn12 = tf.keras.layers.BatchNormalization()
        c12_bn = bn12(c12, training=True)
        m12 = tf.nn.relu(c12_bn, name='conv12_result')

    with tf.variable_scope("deconv3") as scope:
        upsampling3=deconv(m12,[0,0,0,0])
        filter_sets = tf.get_variable('WD3', shape=[3, 3, 3, 3, 2*base_num, 2*base_num], initializer=tf.keras.initializers.he_normal())
        d3 = conv4d(input=upsampling3, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='deconv3')
        dbn3 = tf.keras.layers.BatchNormalization()
        d3_bn = dbn3(d3, training=True)
        dm3 = tf.nn.relu(d3_bn, name='deconv3_result')

    copy3 = tf.concat([m2, dm3], 5)

    with tf.variable_scope("conv13") as scope:
        filter_sets = tf.get_variable('W13', shape=[3, 3, 3, 3, 3*base_num, base_num], initializer=tf.keras.initializers.he_normal())
        c13 = conv4d(input=copy3, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv13')
        bn13 = tf.keras.layers.BatchNormalization()
        c13_bn = bn13(c13, training=True)
        m13 = tf.nn.relu(c13_bn, name='conv13_result')

    with tf.variable_scope("conv14") as scope:
        filter_sets = tf.get_variable('W14', shape=[3, 3, 3, 3, base_num, base_num], initializer=tf.keras.initializers.he_normal())
        c14 = conv4d(input=m13, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv14')
        bn14 = tf.keras.layers.BatchNormalization()
        c14_bn = bn14(c14, training=True)
        m14 = tf.nn.relu(c14_bn, name='conv14_result')

    with tf.variable_scope("conv15") as scope:
        filter_sets = tf.get_variable('W15', shape=[3, 3, 3, 3, base_num, 1], initializer=tf.keras.initializers.he_normal())
        c15 = conv4d(input=m14, filter_sets=filter_sets, strides=[1, 1, 1, 1], name='conv15')
        bn15 = tf.keras.layers.BatchNormalization()
        c15_bn = bn15(c15, training=True)
        m15 = tf.nn.relu(c15_bn, name='conv15_result')

    deconv4 = deconv(m15,[0,0,0,0])
    pooling4 = Maxpooling4d(deconv4, [2, 2, 2, 2], [2, 2, 2, 2])
    input_3d = tf.reshape(pooling4,[1, 48, 56, 48, 88])

    with tf.variable_scope("conv16") as scope:
        filter_sets = tf.get_variable('W16', shape=[3, 3, 3, 88, 44], initializer=tf.keras.initializers.he_normal())
        c16 = tf.nn.conv3d(input_3d, filter_sets, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn16 = tf.keras.layers.BatchNormalization()
        c16_bn = bn16(c16, training=True)
        m16 = tf.nn.relu(c16_bn, name='conv16_result')

    with tf.variable_scope("conv17") as scope:
        filter_sets = tf.get_variable('W17', shape=[3, 3, 3, 44, 44], initializer=tf.keras.initializers.he_normal())
        c17 = tf.nn.conv3d(m16, filter_sets, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn17 = tf.keras.layers.BatchNormalization()
        c17_bn = bn17(c17, training=True)
        m17 = tf.nn.relu(c17_bn, name='conv17_result')

    with tf.variable_scope("conv18") as scope:
        filter_sets = tf.get_variable('W18', shape=[3, 3, 3, 44, 22], initializer=tf.keras.initializers.he_normal())
        c18 = tf.nn.conv3d(m17, filter_sets, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn18 = tf.keras.layers.BatchNormalization()
        c18_bn = bn18(c18, training=True)
        m18 = tf.nn.relu(c18_bn, name='conv18_result')

    with tf.variable_scope("conv19") as scope:
        filter_sets = tf.get_variable('W19', shape=[3, 3, 3, 22, 22], initializer=tf.keras.initializers.he_normal())
        c19 = tf.nn.conv3d(m18, filter_sets, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn19 = tf.keras.layers.BatchNormalization()
        c19_bn = bn19(c19, training=True)
        m19 = tf.nn.relu(c19_bn, name='conv19_result')

    with tf.variable_scope("conv20") as scope:
        filter_sets = tf.get_variable('W20', shape=[3, 3, 3, 22, 1], initializer=tf.keras.initializers.he_normal())
        c20 = tf.nn.conv3d(m19, filter_sets, strides=[1, 1, 1, 1, 1], padding='SAME')
        bn20 = tf.keras.layers.BatchNormalization()
        c20_bn = bn20(c20, training=True)
        m20 = tf.nn.relu(c20_bn, name='conv20_result')

    result = tf.reshape(m20,[48, 56, 48], name='result_s')
    return result

# input
x = tf.placeholder(dtype=tf.float32, shape=[1, 48, 56, 48, 88, 1], name='input_x')
y_s = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='label_s')
train_num = 160
test_num = 40
epoch_num = 150
batch_size = 1

# w_s and w_t needed
predict_s = Xinet(x)
temp = tf.concat([tf.reshape(predict_s,(48, 56, 48, 1)), tf.reshape(y_s,(48, 56, 48, 1))], axis=3)
loss1 = tf.reduce_sum(tf.reduce_min(temp, axis=3, keepdims=False))
loss2 = (tf.reduce_sum(y_s)+tf.reduce_sum(predict_s))/2
loss = -loss1/loss2

lr = 0.0001
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# save
saver = tf.train.Saver(max_to_keep=200)

# begin train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

x_r, y_s_r, test_x_r, test_y_s_r = load_data()
for epoch in tqdm(range(epoch_num)):
    path = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch%s' % epoch
    os.mkdir(path)
    epoch_loss = 0
    val_loss = 0
    for i in range(int((train_num - 1) / batch_size) + 1):
        min_end = min(batch_size * (i + 1), train_num)
        epoch_x, epoch_y_s = x_r[i * batch_size:min_end, :, :, :, :], y_s_r[i * batch_size:min_end,:, :, :].reshape(48, 56, 48)
        sess.run(optimizer, feed_dict={x: epoch_x, y_s: epoch_y_s})
    saver.save(sess, r'/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/Xinet.ckpt-' + str(epoch))
    
    #cal_loss
    for i in range(train_num):
        epoch_x, epoch_y_s = x_r[i, :, :, :, :].reshape(1, 48, 56, 48, 88, 1), y_s_r[i,:, :, :].reshape(48, 56, 48)
        loss_total, test_s = sess.run([loss, predict_s],feed_dict={x: epoch_x, y_s: epoch_y_s})
        epoch_loss += loss_total / train_num
        path = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch%s/%s' %(epoch,i+1)
        os.mkdir(path)
        save1 = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch%s/%s/space.mat' %(epoch,i+1)
        scio.savemat(save1, {'space':test_s})
    print('Epoch: %03d/%03d train_loss: %.9f' % (epoch, epoch_num, epoch_loss))
    f = open('/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result', 'a')
    f.write('\n' + str(epoch_loss))
    f.close()

    for i in range(test_num):
        val_x, val_y_s = test_x_r[i, :, :, :, :].reshape(1, 48, 56, 48, 88, 1), test_y_s_r[i, :, :, :].reshape(48, 56, 48)
        loss_total, test_s = sess.run([loss, predict_s],feed_dict={x: val_x, y_s: val_y_s})
        val_loss += loss_total / test_num
        path = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch%s/%s' %(epoch,train_num+i+1)
        os.mkdir(path)
        save1 = '/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result_mat/epoch%s/%s/space.mat' %(epoch,train_num+i+1)
        scio.savemat(save1, {'space':test_s})
    print('Epoch: %03d/%03d loss_val: %.9f' % (epoch, epoch_num, val_loss))
    f = open('/media/D/alex/NEW_GA4DCNN/medical_image_analysis/spatial_4dcnn/train35/result', 'a')
    f.write('\n' + str(val_loss))
    f.close()

