#!/usr/bin/env python
#coding:utf-8
"""
  Author: Yeliang Li
  Github: (https://github.com/YeliangLi)
  Created: 2018/7/2
"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell,LSTMCell

class Detection(RNNCell):
    def __init__(self,num_units,forget_bias=1.0,activation=None,name=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        if not activation:
            self._activation = tf.nn.tanh
        self._name = name
    
    @property
    def output_size(self):
        return self._num_units,1
    
    @property
    def state_size(self):
        return self._num_units,1
    
    def zero_state(self,batch_size,dtype):
        p_0 = tf.zeros([batch_size,self._num_units],tf.float32)
        r_0 = tf.ones([batch_size,1])
        return p_0,r_0
    
    def __call__(self, inputs, state, scope=None):
        if not self._name:
            name = self.__class__.__name__
        else:
            name = self._name
        with tf.variable_scope(scope or name):
            current_x = inputs[0]
            next_x = inputs[1]
            pre_p = state[0]
            pre_r = state[1]
            i,f,j = tf.split(tf.layers.dense(tf.concat([current_x,pre_p],axis=-1),3*self._num_units),
                             3,axis=-1)
            p_ = tf.nn.sigmoid(f+self._forget_bias) * pre_p + \
                tf.nn.sigmoid(i) * self._activation(j)
            new_p = (1.0 - pre_r) * p_ + \
                     pre_r * tf.layers.dense(current_x,self._num_units,self._activation)
            new_r = tf.layers.dense(tf.concat([new_p,next_x],axis=-1),1,tf.nn.sigmoid)
            outputs = (new_p,new_r)
            next_state = (new_p,new_r)
            return outputs,next_state

class Description(RNNCell):
    def __init__(self,num_units,forget_bias=1.0,activation=None,name=None):
        self._cell = LSTMCell(num_units,forget_bias=forget_bias,activation=activation)
        self._num_units = num_units
        self._name = name
        
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size,dtype)
    
    def __call__(self,inputs,state,scope=None):
        if not self._name:
            name = self.__class__.__name__
        else:
            name = self._name
        with tf.variable_scope(scope or name):
            p = inputs[0]
            r = inputs[1]
            p_c = tf.get_variable("p_constant",shape=[self._num_units,],
                                  dtype=tf.float32,initializer=tf.zeros_initializer())
            m = (1.0 - r) * p_c + r * p
            outputs,new_state = self._cell(m,state)
            return outputs,new_state

    

class SNELSD(RNNCell):
    '''
    reference https://arxiv.org/pdf/1711.05433.pdf
    '''
    def __init__(self,num_units,detection_forget_bias = 1.0,description_forget_bias = 1.0,
                 activation=None,name=None,reuse=None):
        self._detection = Detection(num_units,detection_forget_bias,activation)
        self._description = Description(num_units,description_forget_bias,activation)
        self._name = name
        self._reuse = reuse
    
    @property
    def output_size(self):
        return self._description.output_size
    
    @property
    def state_size(self):
        p_size,r_size = self._detection.state_size
        c_size,h_size = self._description.state_size
        return p_size,r_size,c_size,h_size
    
    def zero_state(self, batch_size, dtype):
        p_0,r_0 = self._detection.zero_state(batch_size,dtype)
        c_0,h_0 = self._description.zero_state(batch_size,dtype)
        return p_0,r_0,c_0,h_0
    
    def __call__(self, inputs, state, scope=None):
        if not self._name:
            name = self.__class__.__name__
        else:
            name = self._name
        with tf.variable_scope(scope or name,reuse=self._reuse):
            pre_p,pre_r,pre_c,pre_h = state
            dection_outs,detection_new_state = self._detection(inputs,(pre_p,pre_r))
            outputs,description_new_state = self._description(dection_outs,(pre_c,pre_h))
            new_state = (detection_new_state[0],detection_new_state[1],
                         description_new_state[0],description_new_state[1])
            return outputs,new_state

    

def gen_SNELSD_inputs(inputs,time_major=False):
    '''
    inputs: (batch_size,time_step,feature_dimension) when time_major = False
    '''
    if time_major:
        inputs = tf.transpose(inputs,[1,0,2])
    batch_size = tf.shape(inputs)[0]
    feature_dim = tf.shape(inputs)[-1]
    next_x = inputs[:,1:,:] #(batch_size,time_step-1,feature_dimension)
    padding = tf.zeros([batch_size,1,feature_dim],tf.float32)
    next_x = tf.concat([next_x,padding],axis=1) 
    if time_major:
        inputs = tf.transpose(inputs,[1,0,2])
        next_x = tf.transpose(next_x,[1,0,2])
    outputs = (inputs,next_x)
    return outputs
    

if __name__ == "__main__":
    '''
    example
    '''
    embedding = tf.Variable(tf.random_normal([20,128]))
    word_ids = tf.placeholder(tf.int32,[None,None])
    sequence_length = tf.placeholder(tf.int32,[None,])
    inputs = tf.nn.embedding_lookup(embedding,word_ids)
    inputs = gen_SNELSD_inputs(inputs)
    cell = SNELSD(256)
    init_state = cell.zero_state(tf.shape(word_ids)[0],tf.float32)
    outputs,_ = tf.nn.dynamic_rnn(cell,inputs,sequence_length,init_state,tf.float32)
    
    #generate data
    x = [[1,4,10,3,5],[8,7,6,0,0]]
    x_seq_len = [5,3]
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(outputs,{word_ids:x,sequence_length:x_seq_len})
        print(o)
        print(o.shape)