import os
import pickle
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Lambda, BatchNormalization, Activation, concatenate, Add, multiply
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from collections import Counter
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

def mySin(x):
    return K.sin(x)

## define a class for Siamese neural network    
class SiameseNetwork(object):
    def __init__(self, args):
        K.clear_session()
        self.input_size = args.input_size
        self.batch_shape = args.batch_shape  ### dimensions of the vetor encoding the batch information
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.step_decay = args.step_decay   ### after how many epochs the learning rate will be subjected to a decay procedure
        self.lr = args.lr
        self.cls_loss = args.cls_loss  ## loss function
        self.beta_1 = args.beta_1
        self.batch = args.batch   ### if or not use batch information for the loss
        self.group = args.group   ### if or not group information for the loss
        self.snet = self._siamese()   ### siamese network for classification
        self.n_group = args.n_group  ### number of groups to classify
        self.cls = self._classifier()  ### ordinary network for classification
        
    def _tune_lr(self, ep):
        if ep%10==0:
            lr = np.clip(self.snet.optimizer.lr*0.75, 1e-7, 1)
            K.set_value(self.snet.optimizer.lr, lr)

    def _get_architecture(self):   ### sub network for the siamese network
        """
        Returns a Convolutional Neural Network based on the input shape given of the images. This is the CNN network
        that is used inside the siamese model. Uses parameters from the siamese one shot paper.
        """
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=11, strides=2, input_shape=self.input_size, name='Conv1' ))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        model.add(Conv1D(filters=64, kernel_size=5, strides=2, name='Conv2'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv1D(filters=128, kernel_size=5, strides=2, name='Conv3'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv1D(filters=128, kernel_size=5, strides=2, name='Conv4'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv1D(filters=64, kernel_size=5, strides=2, name='Conv5'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv1D(filters=64, kernel_size=5, strides=1, name='Conv6'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        model.add(Flatten())
        # model.add(Dense(4096, activation=None))
        # model.add(Activation(mySin))
        model.add(Dense(1024, activation=None))
        model.add(Activation(mySin))
        model.add(Dense(128, activation=None))
        model.add(Activation('relu'))
        return model

    def _siamese(self):   ### define archietecture of siamese network
        model = self._get_architecture()
        
        left_input = Input(self.input_size)
        right_input = Input(self.input_size)
        
        input_batch = Input(self.batch_shape)    ### input of batch encoding
        # in_batch = Dense(16, activation='relu')(input_batch)
        in_batch = Dense(64, activation=None)(input_batch)
        in_batch = Activation(mySin)(in_batch)
        
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        
        output = K.abs(encoded_l - encoded_r)
        output1 = Activation(mySin)(K.mean(K.abs(encoded_l - encoded_r), axis=1))   ### intermediate output, representing the difference between the output of the two sub networks.
        
        output = Dense(64, activation=None)(output)
        output = Activation('relu')(output)
        
        output = Dense(2, activation='sigmoid')(output)

        if self.batch:
            siamese_net = Model(inputs=[left_input, right_input, input_batch], outputs=[output,output1])
        else:
            siamese_net = Model(inputs=[left_input, right_input], outputs=[output,output1])
        siamese_net.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1), 
                            loss=self.cls_loss)
        
        return siamese_net

    def _classifier(self):   #### define architecture of the ordinary neural network
        model = self._get_architecture()
        
        input = Input(self.input_size)
        input_batch = Input(self.batch_shape)
        
        output = model(input)
        output = Dense(64, activation=None)(output)
        output = Activation('relu')(output)       
        output = Dense(self.n_group, activation='sigmoid')(output)

        cls_net = Model(inputs=[input], outputs=output)
        cls_net.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1), 
                            loss=self.cls_loss) 
        return cls_net
        
    def _load_weights(self, weights_file):
        """
        A function that attempts to load pre-existing weight files for the siamese model. If it succeeds then returns
        True and updates the weights, otherwise False.
        :return True if the file is already exists
        """
        # self.siamese_net.summary()
        self.load_file = weights_file
        if os.path.exists(weights_file):  # if the file is already exists, load and return true
            print('Loading pre-existed weights file')
            self.siamese_net.load_weights(weights_file)
            return True
        return False

    def train_snet(self, xL, xR, g, cb, f_model='_weights.h5'):  #### procedure to train the siamese network
        if xL.shape != xR.shape:
            raise Exception("the size of left input and right input does not match!")

        if self.batch and self.group: 
            f_model = 'snet_gb' + f_model
        elif self.group: 
            f_model = 'snet_g' + f_model
        elif self.batch: 
            f_model = 'snet_b' + f_model
        else:
            f_model = 'snet' +f_model
            
        train_size = xL.shape[0]//10*9
        ix_train = np.random.choice(range(xL.shape[0]), train_size, replace=False)
        ix_test = np.array(list(set(range(xL.shape[0])) - set(ix_train)), dtype='int64')

        test_g = np.zeros((len(ix_test), 2))
        test_g[g[ix_test]>0.5, 0] = 1   ### same group
        test_g[g[ix_test]<0.5, 1] = 1
        
        train_size = len(ix_train)
        cnt_cnt = 0
        min_loss = np.inf
        loss_train = []
        loss_eval = []
        progbar = keras.utils.Progbar(self.epochs)       
        for ep in range(self.epochs):
            for ep1 in range(0, train_size-self.batch_size+1, self.batch_size):   
                ix_cur = ix_train[ep1:(ep1+self.batch_size)]
                cur_s1 = xL[ix_cur, :]
                cur_s2 = xR[ix_cur, :]
                
                cur_s1 = np.expand_dims(cur_s1, -1)
                cur_s2 = np.expand_dims(cur_s2, -1)
                
                cur_cb = cb[ix_cur, :]
                gg = g[ix_cur]
                cur_g = np.zeros((len(ix_cur), 2))
                cur_g[gg>0.5, 0] = 1   ### same group
                cur_g[gg<0.5, 1] = 1
                
                with tf.GradientTape() as t_cls:   ### classification loss calculated in different cases
                    if self.batch and self.group:
                        cur_out, cur_out1 = self.snet([cur_s1, cur_s2, cur_cb])
                        loss1 = 0.5*self.cls_loss(cur_cb[:,0], cur_out1) +0.5*self.cls_loss(cur_g[:,1], cur_out1) 
                        cls_loss = 0.5*self.cls_loss(cur_g, cur_out) + 0.5*loss1
                    elif self.batch:
                        cur_out, cur_out1 = self.snet([cur_s1, cur_s2, cur_cb])
                        cls_loss = 0.5*self.cls_loss(cur_g, cur_out) + 0.5*self.cls_loss(cur_cb[:,0], cur_out1) 
                    elif self.group:
                        cur_out, cur_out1 = self.snet([cur_s1, cur_s2])
                        cls_loss = 0.5*self.cls_loss(cur_g, cur_out) + 0.5*self.cls_loss(cur_g[:,1], cur_out1)
                    else:
                        cur_out, cur_out1 = self.snet([cur_s1, cur_s2])
                        cls_loss = self.cls_loss(cur_g, cur_out)
                    grad = t_cls.gradient(cls_loss, self.snet.trainable_variables)
                    self.snet.optimizer.apply_gradients(zip(grad, self.snet.trainable_variables))

            cur_s1 = np.expand_dims(xL[ix_test, :], -1)
            cur_s2 = np.expand_dims(xR[ix_test, :], -1)
            

            ### validation with validation data
            if self.batch and self.group:
                pred, pred1 = self.snet.predict([cur_s1, cur_s2, cb[ix_test, :]], verbose=0)
                loss = 0.5*self.cls_loss(test_g, pred) + 0.5*0.5*(self.cls_loss(test_g[:,1], pred1)+self.cls_loss(cb[ix_test, 0], pred1))
            elif self.batch:
                pred, pred1 = self.snet.predict([cur_s1, cur_s2, cb[ix_test, :]], verbose=0)
                loss = 0.5*self.cls_loss(test_g, pred) + 0.5* self.cls_loss(cb[ix_test, 0], pred1)
            elif self.group:
                pred, pred1 = self.snet.predict([cur_s1, cur_s2], verbose=0)
                loss = 0.5*self.cls_loss(test_g, pred) + 0.5* self.cls_loss(test_g[:,1], pred1)
            else:
                pred, pred1 = self.snet.predict([cur_s1, cur_s2], verbose=0)
                loss = self.cls_loss(test_g, pred)

            if ep<50:
                self.snet.save_weights(f_model, overwrite=True)
            elif ep>=50:     ### only save weights if there are further improvement in the validation 
                if min_loss>loss+1e-3:
                    min_loss = loss
                    cnt_cnt = 0
                    self.snet.save_weights(f_model, overwrite=True)
                else:
                    cnt_cnt += 1
                    
                if cnt_cnt>=10:
                    self.snet.load_weights(f_model)
                    break

            loss_train = np.append(loss_train, cls_loss)
            loss_eval = np.append(loss_eval, loss)
            progbar.add(1, values=[("train_loss", cls_loss.numpy()), ("eval_loss", loss.numpy())])
            
        self.snet.load_weights(f_model)    #### load the weights giving the best performance according to the validation
        return loss_train, loss_eval

    #### prediction with siamese network 
    def pred_snet(self, x_train, label_train, x_test, labels_test, batches_test):
        n_test = x_test.shape[0]
        n_train = x_train.shape[0]
        x_train = np.expand_dims(x_train, -1)
        pred_test = []
        for i in range(n_test):
            latent = tf.tile(x_test[i, :][np.newaxis,:], [n_train, 1])
            latent = np.expand_dims(latent, -1)
            if self.batch:   ### if batch information is used
                cb = []
                cb1 = []
                for j in range(n_train):
                    g_tmp1 = np.asarray([np.random.uniform(0.9, 1.0) for ii in range(self.batch_shape)])
                    cb1.append(g_tmp1)
                    g_tmp = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(self.batch_shape)])
                    cb.append(g_tmp)    
                cb = np.row_stack(cb)
                cb1 = np.row_stack(cb1)  
                
                b_eval, _ = self.snet([x_train, latent, cb])
                b_eval = b_eval.numpy()
                b_eval1, _ = self.snet([x_train, latent, cb1])
                b_eval1 = b_eval1.numpy()

                b_eval = np.concatenate([b_eval, b_eval1], axis=1)
                b_eval = np.argmax(b_eval, axis=1)%2
                b_eval = np.array(b_eval, dtype='int32')
            else:
                b_eval, _ = self.snet([x_train, latent])
                b_eval = np.argmax(b_eval, axis=1)

            cur_pred = label_train[(b_eval==0)]
                
            vote = Counter(cur_pred).most_common(1)
            if len(vote)<1:
                pred_test = np.append(pred_test, 'nan')
            else:
                pred_test = np.append(pred_test, vote[0][0])


        #### calculate benchmarks for the prediction
        acc = []
        min_acc = []
        std_acc = []
        method = []
        b_test = []
        for ll in np.unique(batches_test):
            i_b = np.argwhere(batches_test==ll)[:,0]
            cc = confusion_matrix(labels_test[i_b], pred_test[i_b], labels=np.unique(label_train))
            accs_all = np.diag(cc)/np.sum(cc, 1)
            min_acc = np.append(min_acc, np.nanmin(accs_all))
            std_acc = np.append(std_acc, np.nanstd(accs_all))
            acc = np.append(acc, np.nanmean(accs_all))
            b_test = np.append(b_test, ll)
            method = np.append(method, 'snet')    
            
        return pred_test, acc, min_acc, std_acc, method, b_test


    ### train ordinary neural network for the classification
    def train_cls(self, x, g, f_model='_weights.h5'):

        f_model = 'cls' + f_model
        
        train_size = x.shape[0]//10*9
        ix_train = np.random.choice(range(x.shape[0]), train_size, replace=False)
        ix_test = np.array(list(set(range(x.shape[0])) - set(ix_train)), dtype='int64')
        
        train_size = len(ix_train)
        cnt_cnt = 0
        min_loss = np.inf
        progbar = keras.utils.Progbar(self.epochs)       
        for ep in range(self.epochs):
            for ep1 in range(0, train_size-self.batch_size+1, self.batch_size):   
                ix_cur = ix_train[ep1:(ep1+self.batch_size)]
                cur_s1 = x[ix_cur, :]
                
                cur_s1 = np.expand_dims(cur_s1, -1)
                
                cur_g = g[ix_cur, :]
                
                with tf.GradientTape() as t_cls:
        
                    cur_out = self.cls([cur_s1])   
                    cls_loss = self.cls_loss(cur_g, cur_out)
    
                    grad = t_cls.gradient(cls_loss, self.cls.trainable_variables)
                    self.cls.optimizer.apply_gradients(zip(grad, self.cls.trainable_variables))

            cur_s1 = np.expand_dims(x[ix_test, :], -1) 
            pred = self.cls([cur_s1])
            loss = self.cls_loss(g[ix_test, :], pred) #self.cls.evaluate([cur_s1], g[ix_test, :], verbose=0)

            if ep<50:
                self.cls.save_weights(f_model, overwrite=True)
            elif ep>=50:
                if min_loss>loss+1e-3:
                    min_loss = loss
                    cnt_cnt = 0
                    self.cls.save_weights(f_model, overwrite=True)
                else:
                    cnt_cnt += 1
                    
                if cnt_cnt>=10:
                    self.cls.load_weights(f_model)
                    break
                    
            progbar.add(1, values=[("train_loss", cls_loss.numpy()), ("eval_loss", loss)])  
            
        self.cls.load_weights(f_model)
                

    #### prediction with ordinary neural network
    def pred_cls(self, x_test, labels_test, batches_test, uni_labels):
        n_test = x_test.shape[0]
        ix_test = np.array(range(n_test))
        pred_test = []
        for i in range(n_test):   
            ix_cur = ix_test[i]
            cur_s1 = x_test[ix_cur, :][np.newaxis,:]
            cur_s1 = np.expand_dims(cur_s1, -1)
            cur_out = self.cls([cur_s1])   
            pred_test = np.append(pred_test, uni_labels[np.argmax(cur_out)])

        acc = []
        min_acc = []
        std_acc = []
        method = []
        b_test = []
        for ll in np.unique(batches_test):
            i_b = np.argwhere(batches_test==ll)[:,0]
            cc = confusion_matrix(labels_test[i_b], pred_test[i_b], labels=uni_labels)
            accs_all = np.diag(cc)/np.sum(cc, 1)
            min_acc = np.append(min_acc, np.nanmin(accs_all))
            std_acc = np.append(std_acc, np.nanstd(accs_all))
            acc = np.append(acc, np.nanmean(accs_all))
            b_test = np.append(b_test, ll)
            method = np.append(method, 'snet')    
            
        return pred_test, acc, min_acc, std_acc, method, b_test


### define loss functions
class loss:
    def __init__(self, alp=0.9):
        self.alp = alp
        
    def loss_bce(self, y_true, y_pred):
        bce = tf.losses.BinaryCrossentropy(from_logits=False)
        loss = bce(y_true, y_pred)
        
        return loss
    
    def loss_ase(self, y_true, y_pred):
        ase = tf.losses.MeanAbsoluteError()
        loss = ase(y_true, y_pred)
        return loss
    
    def mse_loss(self, y_true, y_pred):
        r_loss = K.mean(K.square(y_true - y_pred))
        return r_loss
    
    def loss_kl(self, mean, log_var):
        kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
        return K.mean(kl_loss)
    
    def loss_l2(self, latent):
        l2_loss =  K.sqrt(K.sum(K.square(latent), axis = 1))
        return K.mean(l2_loss)