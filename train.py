import numpy as np #supporting multi-dimensional arrays and matrices
import os #read or write a file
import cv2  
import pandas as pd #data manipulation and analysis
from tqdm import tqdm # for  well-established ProgressBar
from random import shuffle
import tools
from getFeature import create_train_data, create_test_data, label_return
LR = 1e-3
# MODEL_NAME = 'plantclassfication-{}-{}.model'.format(LR, '2conv-basic')

data_dir = 'F:/dataset/all/'
train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
IMG_SIZE = 224
NUM_CLASS = 12
LEARNING_RATE_BASE = 0.0008
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.00004
MODEL_SAVE_PATH="plant_model/"
MODEL_NAME="plant_model"
BATCH_SIZE = 32
MAX_STEPS = 15000
MOVING_AVERAGE_DECAY = 0.99
NUN_THREADS = 4
min_after_dequeue = 1000
capacity = min_after_dequeue + 3 * BATCH_SIZE

def get_batchs(data, BATCH_SIZE):
    size = data.shape[0]
    for i in range(size//BATCH_SIZE):
        if (i+1)*BATCH_SIZE > size:
            yield data[i*BATCH_SIZE:]
        else:
            yield data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            
import tensorflow as tf
tf.reset_default_graph()

def VGG16PlanInferencet(x, n_classes, is_pretrain=True): 
    x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            

    x = tools.FC_layer('fc6', x, out_nodes=4096)
    x = tools.batch_norm(x)
    x = tools.FC_layer('fc7', x, out_nodes=4096)
    x = tools.batch_norm(x)
    x = tools.FC_layer('fc8', x, out_nodes=n_classes)

    return x
# 
# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('model loaded!')


if __name__ == '__main__':
    train = create_train_data()
    testData = create_test_data()
    with tf.name_scope("input"):  
        x_image = tf.placeholder(tf.float32, shape=[None,IMG_SIZE,IMG_SIZE,3], name = 'x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, 12], name = 'y-input') 
        
        X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
        Y = [i[1] for i in train]  
        test_x = np.array([i[0] for i in testData]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
        test_y = [i[1] for i in testData]
#         keep_prob = tf.placeholder(tf.float32)      
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y_conv = VGG16PlanInferencet(x_image, NUM_CLASS)
    global_step = tf.Variable(0, trainable=False)    
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
 
    with tf.name_scope("loss_average"):    
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) 
#         loss = cross_entropy + regularizer       
#         loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        loss = cross_entropy
        
    with tf.name_scope("train_step"):         
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 15000, LEARNING_RATE_DECAY, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)     
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')           
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("/path/to//log", tf.get_default_graph())
    writer.close()  

    with tf.Session() as sess:    
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        for i in range(20000//BATCH_SIZE):
            image_batch, label_batch = tf.train.shuffle_batch([X, Y], 
                                                      batch_size = BATCH_SIZE, 
                                                      capacity = capacity, 
                                                      min_after_dequeue = min_after_dequeue, 
                                                      num_threads = NUN_THREADS)  
            if i%10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x_image: X, y_: Y})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            if i%100 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step= global_step)
#                 print("test accuracy %g"%accuracy.eval(feed_dict={
#                                                       x: validation.images, y_: validation.labels, keep_prob: 1.0}))          
            train_step.run(feed_dict={x_image: X, y_: Y})
        coord.request_stop()
        coord.join(threads)
     
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path: saver.restore(sess, ckpt.model_checkpoint_path)
        for data in testData:
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
            prediction = tf.argmax(y_conv, 1)
            test_labels = prediction.eval(feed_dict={x_image: data})
            
        
    
# model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
#     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# model.save(MODEL_NAME)

    
# import matplotlib.pyplot as plt
# test_data = create_test_data()
# fig=plt.figure(figsize = (18,10))
# for num,data in enumerate(test_data[:12]): 
#     img_num = data[1]
#     img_data = data[0]
#     y = fig.add_subplot(3,4,num+1)
#     orig = img_data
#     data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#     model_out = model.predict([data])[0]
#     str_label=label_return (model_out)
#     y.imshow(orig,cmap='gray',interpolation='nearest')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# plt.show()

test_data = create_test_data()
with open('sample_submission.csv','w') as f:
    f.write('file,species\n')
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
        model_out = model.predict([data])[0]
        str_label=label_return (model_out)
        file = img_num
        species = str_label
        row = file + "," + species + "\n"
        f.write(row)



