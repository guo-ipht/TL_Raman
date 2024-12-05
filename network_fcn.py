# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:08:07 2022

@author: user
"""

import os
import numpy as np 
import argparse
import glob
import tensorflow as tf
from skimage import io
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tensorflow.keras import layers
from keras.layers import concatenate
from keras.initializers import glorot_uniform
from scipy.stats import entropy
from tensorflow import keras
import tensorflow.keras.backend as K
import pickle
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
from collections import Counter

def mySin(x):
    return K.square(K.sin(x))

def foo(v):
    if v<0 : return K.sin(v)
    else: return v
    
def reluSin(x):
    return K.sin(x)
    

# A constum layer to add random noise
class Noise(keras.layers.Layer):
    def __init__(self, mean=0, stddev=1.0, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)
        self.mean = mean
        self.stddev = stddev

    def call(self, inputs):
        return inputs + tf.random.normal(
                inputs.shape[1:], 
                mean=self.mean,
                stddev=self.stddev
            ) 
        
#### class of variational autoencoder
class model_vae(object):
    def __init__(self, args):
        self.input_size = args.input_size
        self.latent_shape = args.latent_shape   ### dimension of the latent variable
        self.para_shape = args.para_shape   ### dimension of the control vector, i.e., if the output spectra are from the same batch/group as the input spectra
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.step_decay = args.step_decay   ### after how many epochs to apply a decay procedure on learning rate.
        self.patience = args.patience    ### patient for early stop
        self.lr = args.lr
        self.dec_loss = args.dec_loss   ### loss function for decoder
        self.beta_1 = args.beta_1
        self.norm = args.norm       
        self.n_down = args.n_down   ### number of downsampling layers in encoder
        self.n_std = args.n_std
        self.enc = self.encoder()   ### encoder
        self.dec = self.decoder()   ### decoder
        self.wn = []    ### reserved parameter for wavenumber 
        self.x = []   ### reserved parameter for input
        self.y = []   ### reserved parameter for output
        # self.c = []
        self.g = []   ### reserved parameter for group
    
    def _get_norm_layer(self):
        if self.norm == 'none':
            return lambda: lambda x: x
        elif self.norm == 'batch_norm':
            return keras.layers.BatchNormalization
        # elif self.norm == 'instance_norm':
        #     return tfa.layers.InstanceNormalization
        elif self.norm == 'layer_norm':
            return keras.layers.LayerNormalization

    ### resampling based on encoder output        
    def sampling(self, input_1=None, input_2=None):
        if input_1 is None:
            input_1 = keras.Input(shape=self.latent_shape)
        if input_2 is None:
            input_2 = keras.Input(shape=self.latent_shape) 
            
        epsilon = K.random_normal(shape=K.shape(input_1), mean=0., stddev=1.)
        z = input_1 + K.exp(input_2 / 2) * epsilon
        # z = layers.Activation('tanh')(z)
        return z

    def _tune_lr(self, ep):
        if ep%5==0:
            lr = np.clip(self.enc.optimizer.lr*0.95, 1e-7, 1)
            K.set_value(self.enc.optimizer.lr, lr)
            K.set_value(self.dec.optimizer.lr, lr)

    ### archetecture of encoder          
    def encoder(self):
        Norm = self._get_norm_layer()
        inputs = keras.Input(shape=self.input_size)
        nb_win = 5

        x = layers.Conv1D(filters=64, kernel_size=nb_win*2+1, strides=2, padding='same', activation=None)(inputs)
        x = layers.Activation('relu')(x)
        x = Norm()(x)
        
        x = layers.Conv1D(filters=128, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)

        for i in range(self.n_down-4):
            x = layers.Conv1D(filters=256, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
            x = layers.Activation('relu')(x)
            x = Norm()(x)
        
        x = layers.Conv1D(filters=128, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)
         
        x = layers.Conv1D(filters=64, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation=None)(x)
        x = layers.Activation('relu')(x)
        
        mean = layers.Dense(self.latent_shape, activation=None)(x)
        mean = layers.Activation('relu')(mean)
        log_var = layers.Dense(self.latent_shape, activation=None)(x)
        log_var = layers.Activation('relu')(log_var)
        
        
        model = keras.Model(inputs, [mean, log_var])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1), 
                      loss=self.dec_loss)
        return model
    

    ### architecture of decoder
    def decoder(self):
        Norm = self._get_norm_layer()
        nb_win = 5
        
        inputs1 = keras.Input(self.latent_shape)
        inputs2 = keras.Input(shape=6)    ### control vector for the generation
        
        x = layers.Dense(self.input_size[0]//(2**self.n_down)*16, activation=None)(inputs1)
        x = layers.Activation('relu')(x)
        x = layers.Reshape((self.input_size[0]//(2**self.n_down), 16))(x)

        x2 = layers.Dense(64, activation=None)(inputs2)
        x2 = layers.Activation(mySin)(x2)
        x2 = layers.Dense(self.input_size[0]//(2**self.n_down)*1, activation=None)(x2)
        x2 = layers.Activation(mySin)(x2)
        x2 = layers.Reshape((self.input_size[0]//(2**self.n_down), 1))(x2)
        
        x = layers.multiply([x, x2])
        
        x = layers.Conv1DTranspose(filters=64, kernel_size=nb_win, strides=1, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)

        x = layers.Conv1DTranspose(filters=128, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)

        for i in range(self.n_down-4):
            x = layers.Conv1DTranspose(filters=256, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
            x = layers.Activation('relu')(x)
            x = Norm()(x)
        
        x = layers.Conv1DTranspose(filters=128, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)
        
        x = layers.Conv1DTranspose(filters=64, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)
        x = Norm()(x)
        
        x = layers.Conv1DTranspose(filters=1, kernel_size=nb_win, strides=2, padding='same', activation=None)(x)
        x = layers.Activation('relu')(x)

        Out = Noise(stddev = self.n_std)(x)[:,:,0]
        
        model = keras.Model(inputs=[inputs1, inputs2], outputs=Out)
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1), 
                      loss=self.dec_loss)
        return model
        
    ### train the VAE network
    def train_vae(self, ix_train, ix_test, ratio=1e-4, f_model='_weights.h5'):
        ### ix_train: index of training data from self.x
        ### ix_test: index of testing data from self.x
        ### ratio: ratio to be used on the KL loss
        ### f_model: ending of the file name for the network weights

        n_org = len(self.wn)
        train_size = len(ix_train)
        downsampling = 2**self.n_down

        f_model1 = 'enc'+ f_model
        f_model2 = 'dec'+ f_model
        
        cnt_cnt = 0
        min_loss = np.inf
        progbar = keras.utils.Progbar(self.epochs)   
        for ep in range(self.epochs):
            if (ep>=self.step_decay):
                self._tune_lr(ep)
                  
            for ep1 in range(0, train_size-self.batch_size+1, self.batch_size):
                    
                ix_cur = ix_train[ep1:(ep1+self.batch_size)]
                    
                spec_in = self.x[ix_cur, :]
                spec_out = self.y[ix_cur, :]
                g_in_out = self.g[ix_cur, :]
                spec_in = np.expand_dims(spec_in, axis=-1)
                
                with tf.GradientTape() as t_en, tf.GradientTape() as t_de: #, tf.GradientTape() as t_dis:
                    mean, log_var = self.enc(spec_in)
                    latent = self.sampling(mean, log_var)
                    y_rec = self.dec([latent, g_in_out])
        
                    enc_loss = self.dec_loss(spec_out, y_rec, mean, log_var, latent, 100, ratio)     
                    dec_loss = enc_loss
                    
                    grad_enc = t_en.gradient(enc_loss, self.enc.trainable_variables)
                    self.enc.optimizer.apply_gradients(zip(grad_enc, self.enc.trainable_variables))
                            
                    grad_dec = t_de.gradient(dec_loss, self.dec.trainable_variables)
                    self.dec.optimizer.apply_gradients(zip(grad_dec, self.dec.trainable_variables))

            spec_in = self.x[ix_test, :]
            spec_out = self.y[ix_test, :]
            g_in_out = self.g[ix_test, :]
        
            spec_in = np.expand_dims(spec_in, axis=-1)
        
            mean, log_var = self.enc(spec_in)
            latent = self.sampling(mean, log_var)
            y_rec = self.dec([latent, g_in_out])
            val_loss = self.dec_loss(spec_out, y_rec, mean, log_var, latent, 100, ratio) 
            
            progbar.add(1, values=[("train_loss", enc_loss.numpy()), ("eval_loss", val_loss.numpy())])

            if ep<50:
                self.enc.save_weights(f_model1, overwrite=True)
                self.dec.save_weights(f_model2, overwrite=True)
            elif ep>=50:     ### early stop
                if min_loss>val_loss+1e-3:
                    min_loss = val_loss
                    cnt_cnt = 0
                    self.enc.save_weights(f_model1, overwrite=True)
                    self.dec.save_weights(f_model2, overwrite=True)
                else:
                    cnt_cnt += 1
                    
                if cnt_cnt>=10:
                    self.enc.load_weights(f_model1)
                    self.dec.load_weights(f_model2)
                    break
                    
        self.enc.load_weights(f_model1)
        self.dec.load_weights(f_model2) 

    ### generate spectra using the trained network       
    def gen_spec(self, x, paras, n_gen=60):
        ### x: input spectra
        ### paras: a collection of values benchmarking the spectral similarity between input and expected output. The generation will be performed using a random value out of this 
        ### n_gen: how many spectra will be generated for each input spectrum
        
        n_org = len(self.wn)
        downsampling = 2**self.n_down
        ix_sb_sg = (self.g[:, 2]>0.5) & (self.g[:, -1]>0.5)
        ix_db_sg = (self.g[:, 2]<0.5) & (self.g[:, -1]>0.5)
        ix_dg = (self.g[:, -1]<0.5)
        
        progbar = keras.utils.Progbar(x.shape[0])
        
        y_gen = []; c_gen = []; c_rea = []; g_gen=[]; b_gen=[]
        
        pool = np.array(range(self.g.shape[0]))

        for i in range(x.shape[0]):
            progbar.add(1)
            spec_in = np.expand_dims(x[i,:], axis=0)
            spec_in = np.expand_dims(spec_in, axis=-1)
            for j in range(n_gen):
                g_in = np.zeros((self.g.shape[1]))
                ixx = np.random.choice(np.array(range(len(paras))), 1)[0]
                pp = random.uniform(0.99, 1.01)
                c0 = paras[ixx]*pp
                c1 = paras[ixx]*(2-pp)
                if j%4==0:  ### different group, different batch
                    g_in[2:] = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(4)]) 
                elif j%4==1:  ### different batch, same group
                    bb = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
                    gg = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
                    g_in[2:] = np.append(bb, gg) 
                elif j%4==2:  ### same batch, same group
                    g_in[2:] = np.asarray([np.random.uniform(0.9, 1) for ii in range(4)]) 
                else:    #### same batch, differet group
                    bb = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
                    gg = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
                    g_in[2:] = np.append(bb, gg) 
                    
                g_in[:2] = [c0**8, c1**8]
                g_in = np.expand_dims(g_in, axis=0)
                
                mean, log_var = self.enc(spec_in)
                latent = self.sampling(mean, log_var)
                y_rec = self.dec([latent, g_in])
                y_rec = y_rec.numpy()[0,:]
                c_rec =  0.5*(1-spatial.distance.cosine(y_rec[:n_org], x[i,:n_org]) + np.corrcoef(y_rec[:n_org], x[i,:n_org])[0, 1])
                    
                y_gen.append(y_rec)
                c_gen = np.append(c_gen, c_rec)
                g_gen = np.append(g_gen, g_in[0, -1])
                b_gen = np.append(b_gen, g_in[0, 2])
                c_rea = np.append(c_rea, (c0+c1)*0.5)
                
        y_gen = np.row_stack(y_gen)  
        
        return y_gen, c_rea, c_gen, g_gen, b_gen

    ### generate spectra using the trained network 
    ### different to gen_spec, here only the similarity is specified directly (no random sampling)
    def gen_cors(self, x, cors):
        n_org = len(self.wn)
        downsampling = 2**self.n_down
        
        progbar = keras.utils.Progbar(x.shape[0])
        
        y_gen = []; c_gen = []; c_rea = []; g_gen=[]; b_gen=[]
        
        pool = np.array(range(self.g.shape[0]))

        for i in range(x.shape[0]):
            progbar.add(1)
            spec_in = np.expand_dims(x[i,:], axis=0)
            spec_in = np.expand_dims(spec_in, axis=-1)
            for j in range(len(cors)):
                pp = random.uniform(0.99, 1.01)
                c0 = cors[j]*pp
                c1 = cors[j]*(2-pp)
                b_in1 = random.choice([random.uniform(0.01, 0.1), random.uniform(0.9, 1)])
                g_in1 = random.choice([random.uniform(0.01, 0.1), random.uniform(0.9, 1)])
                g_in = np.zeros((self.g.shape[1]))
                if b_in1<0.5 and g_in1<0.5:      ### different group, different batch
                    g_in[2:] = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(4)]) 
                elif b_in1<0.5 and g_in1>0.5:    ### different batch, same group
                    bb = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
                    gg = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
                    g_in[2:] = np.append(bb, gg) 
                elif b_in1>0.5 and g_in1>0.5:   ### same batch, same group
                    g_in[2:] = np.asarray([np.random.uniform(0.9, 1) for ii in range(4)]) 
                else:    #### same batch, differet group
                    bb = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
                    gg = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
                    g_in[2:] = np.append(bb, gg) 

                g_in[:2] = [c0**8, c1**8]
                
                g_in = np.expand_dims(g_in, axis=0)
                
                mean, log_var = self.enc(spec_in)
                latent = self.sampling(mean, log_var)
                y_rec = self.dec([latent, g_in])
                y_rec = y_rec.numpy()[0,:]
                c_rec =  0.5*(1-spatial.distance.cosine(y_rec[:n_org], x[i,:n_org]) + np.corrcoef(y_rec[:n_org], x[i,:n_org])[0, 1])
                    
                y_gen.append(y_rec)
                c_gen = np.append(c_gen, c_rec)
                g_gen = np.append(g_gen, g_in[0, -1])
                b_gen = np.append(b_gen, g_in[0, 2])
                c_rea = np.append(c_rea, cors[j])
                
        y_gen = np.row_stack(y_gen)  
        
        return y_gen, c_rea, c_gen, g_gen, b_gen

    ### generate spectra using the trained network 
    ### different to gen_spec, here only the similarity is specified directly (no random sampling) 
    def gen_mt(self, x, cors):
        n_org = len(self.wn)
        downsampling = 2**self.n_down
        
        progbar = keras.utils.Progbar(x.shape[0])
        
        y_gen = []; c_gen = []; c_rea = []; g_gen=[]; b_gen=[]
        
        pool = np.array(range(self.g.shape[0]))

        for i in range(x.shape[0]):
            spec_in = np.expand_dims(x[i,:], axis=0)
            spec_in = np.expand_dims(spec_in, axis=-1)
            progbar.add(1)
            for j in range(len(cors)):
                pp = random.uniform(0.99, 1.01)
                c0 = cors[j]*pp
                c1 = cors[j]*(2-pp)
                for k in range(4):
                    g_in = np.zeros((self.g.shape[1]))
                    if k==0:
                        g_in[2:] = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(4)]) 
                    elif k==1:
                        bb = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
                        gg = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
                        g_in[2:] = np.append(bb, gg)
                    elif k==2:
                        g_in[2:] = np.asarray([np.random.uniform(0.9, 1) for ii in range(4)]) 
                    else:
                        bb = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
                        gg = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
                        g_in[2:] = np.append(bb, gg) 
    
                    g_in[:2] = [c0**8, c1**8]
                    
                    g_in = np.expand_dims(g_in, axis=0)
                    
                    mean, log_var = self.enc(spec_in)
                    latent = self.sampling(mean, log_var)
                    y_rec = self.dec([latent, g_in])
                    y_rec = y_rec.numpy()[0,:]
                    c_rec =  0.5*(1-spatial.distance.cosine(y_rec[:n_org], x[i,:n_org]) + np.corrcoef(y_rec[:n_org], x[i,:n_org])[0, 1]) 
                        
                    y_gen.append(y_rec)
                    c_gen = np.append(c_gen, c_rec)
                    g_gen = np.append(g_gen, g_in[0, -1])
                    b_gen = np.append(b_gen, g_in[0, 2])
                    c_rea = np.append(c_rea, cors[j])
                
        y_gen = np.row_stack(y_gen)  
        
        return y_gen, c_rea, c_gen, g_gen, b_gen

### loss function
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
     
    def loss_vae(self, y_true, y_pred, mean, log_var, latent, coef1=100, coef2=0.001):
        r_ls = self.custom_loss(y_true, y_pred)
        kl_ls = self.loss_kl(mean, log_var)
        return  coef1*r_ls + coef2*kl_ls 

    def custom_loss(self, y_true, y_pred):
        # return self.loss_bce(y_true, y_pred)+ 0.5*self.cos_dist(y_true, y_pred)
        diff = y_true-y_pred
        # return K.mean(K.abs(diff)) + K.mean(K.std(diff, axis=1)) # self.cos_dist(y_true, y_pred)
        return K.mean(K.abs(diff)) #+ 0.25*K.mean(K.std(diff, axis=1)) #+ 0.5*K.mean(K.std(diff, axis=0))

    def cos_dist(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        n_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=1))
        n_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1))
        denom = tf.multiply(n_true, n_pred)
        num = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)
        return 1-tf.reduce_mean(tf.divide(num, denom))