# Copyright (c) 2020 smarsu. All Rights Reserved.

import smnet as sm
import tensorflow as tf

import evaluation


class TestBase(object):
  def __init__(self, op_name, sm_func, base_func, inputs_func, lr=0.1, momentum=0.9, weight_decay=5e-4, epoch=200, loc=0., scale=1., bp=True):
    self._op_name = op_name
    self._base_func = base_func
    self._sm_func = sm_func
    self._inputs_func = inputs_func
    self._lr = lr
    self._momentum = momentum
    self._weight_decay = weight_decay
    self._epoch = epoch
    self._loc = loc
    self._scale = scale
    self._bp = bp

    self._test_time = 0

  
  def test_case(self, **params):
    print("---------------- {} test{} ----------------".format(self._op_name, self._test_time))
    self._test_time += 1

    self._args = self._inputs_func(**params, loc=self._loc, scale=self._scale)
    sm_fst_res, sm_lst_res = self._run_sm_func()
    base_fst_res, base_lst_res = self._run_base_func()

    assert sm_fst_res.shape == sm_lst_res.shape == base_fst_res.shape == base_lst_res.shape, \
      '{} {} {} {}'.format(sm_fst_res.shape, sm_lst_res.shape, base_fst_res.shape, base_lst_res.shape)

    print('compare first result:')
    evaluation.compare_all(sm_fst_res.flatten(), base_fst_res.flatten())
    print('compare last result:')
    evaluation.compare_all(sm_lst_res.flatten(), base_lst_res.flatten())
    print()

  
  def _run_sm_func(self):
    opt = sm.SGD(lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)
    args = self._args
    for i in range(self._epoch):
      res, args = self._sm_func(*args)
      if i == 0:
        fst_res = res.data.copy()
      
      opt.minimum(res)
    
    lst_res, _ = self._sm_func(*args)
    lst_res = lst_res.data.copy()

    return fst_res, lst_res


  def _run_base_func(self):
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
      args = self._args
      y, _ = self._base_func(*args)

      loss = tf.reduce_sum(y)
      if self._weight_decay != 0:
        for variable in tf.global_variables():
          loss += tf.reduce_sum(self._weight_decay * tf.nn.l2_loss(variable))
    
      if self._bp:
          opt = tf.train.MomentumOptimizer(self._lr, self._momentum).minimize(loss)
      
      sess.run(tf.global_variables_initializer())
      
      for i in range(self._epoch):
        res = sess.run(y)
        if i == 0:
          fst_res = res

        sess.run(opt)

      lst_res = sess.run(y)

      return fst_res, lst_res
