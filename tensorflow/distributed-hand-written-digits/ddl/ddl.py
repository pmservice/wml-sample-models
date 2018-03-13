# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2018. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

# Integration of DDL in Tensorflow scripts

# This Python module exports the following functions:
# - num_hosts(): number of hosts deployed
# - rank(): MPI rank number assigned to this process
# - size(): total number of MPI ranks (= number of GPUs)
# - local_rank(): id of GPU used by this process

# - init(options): initialize this module
# - bcast(init_val): broadcast initial value
# - allreduce_n(grads): DDL-reduces all gradient tensors in one go
# - grads_reduce(grads_and_vars):

import os
import tensorflow as tf

# Design decision: get all MPI data via env NOT from ddl_MDR.init()!

mpi_init_status = True

# Get some MPI environment variables values:
mpi_vars = { 'OMPI_COMM_WORLD_SIZE':0,
             'OMPI_COMM_WORLD_LOCAL_SIZE':0,
             'OMPI_COMM_WORLD_RANK':0,
             'OMPI_COMM_WORLD_LOCAL_RANK':0 }

# Ensure all the necessary ENV variables are available;
# else don't initialize DDL & restrict DDL API invocations.
for var in mpi_vars:
    val = os.getenv(var)
    if val is None:
        mpi_init_status = False
        break;
    mpi_vars[var] = val

if mpi_init_status:
    #Get as number:
    world_size = int(mpi_vars['OMPI_COMM_WORLD_SIZE'])
    local_size = int(mpi_vars['OMPI_COMM_WORLD_LOCAL_SIZE'])
    world_rank = int(mpi_vars['OMPI_COMM_WORLD_RANK'])
    rank_local = int(mpi_vars['OMPI_COMM_WORLD_LOCAL_RANK'])

    DDL_OPTIONS="-mode p:1x2 -dbg_level 1 -dump_iter 50"
    # Get any options to configure DDL:
    #DDL_OPTIONS = os.getenv('DDL_OPTIONS')
    if DDL_OPTIONS is None:
        print('DDL: DDL_OPTIONS is not set; need explicit init(options).')
else:
    world_size = 1
    local_size = 1
    world_rank = 0
    rank_local = 0

# The number of nodes (or hosts)
def num_hosts():
    return world_size // local_size

# rank runs from 0 to size (excl)
# uniquely identifies each parallel process
def rank():
    return world_rank

# size is the total number of GPUs (MPI processes)
# typically product of number of nodes and GPUs per node
def size():
    return world_size

# local_rank runs from 0 to the number of GPUs (excl) per node
# number of GPUs typically the same for each node, say 4
def local_rank():
    return rank_local

_tf_Session = tf.Session.__init__

def new_init(session_object, target='', graph=None, config=None):
    # Echo all info so far using our own API:
    # mpi info and assigned GPU:
    print('DDL: rank: {}, size: {}, gpuid: {}, hosts: {}'.format(
        rank(), size(), local_rank(), num_hosts()))

    if config is None:
        config = tf.ConfigProto()

    # Weird TF behavior: without allow_growth set here, TF will already
    # here allocate all memory on the GPU; even if later sessions use
    # a config that also sets allow_growth.
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(local_rank())

    _tf_Session(session_object, target, graph, config)

# Session redefined, if required
if mpi_init_status:
    tf.Session.__init__ = new_init

ddl_MDR=None
def _init(options):
    """A function which initializes DDL.
    """
    # Dynamically load DDL TF library:
    global ddl_MDR
    if ddl_MDR is not None:
        raise RuntimeError('DDL has already been initialized.')
    ddl_MDR = tf.load_op_library('ddl_MDR.so')

    # Initialize DDL:
    with tf.Session() as sess:
        sess.run(ddl_MDR.init(local_rank(), mode = options))
    print('DDL: module {} initialized.'.format(rank()))

_prev_bcast_op = None
# Creates broadcast node to share MPI root initial value among workers.
# Inserts control dependency on global prev_bcast_op under the covers.
# Returns initial value (Tensor) of root.
def _bcast(init_val):
    global _prev_bcast_op
    if _prev_bcast_op is None: # first time
        init_val = ddl_MDR.bcast(init_val)
    else:
        with tf.control_dependencies([_prev_bcast_op]):
            init_val = ddl_MDR.bcast(init_val)
    _prev_bcast_op = init_val
    return init_val

# Intercept tf.Variable constructor:
_tf_Var = tf.Variable.__init__

# Our replacement:
# Insert a Bcast operator that will provide the initial value.
# Mind to enforce a certain consistent execution order across multiple instances.
def newVar(self, *args, **kwargs):
  #print("rank: {}, kwargs: {}".format(rank, kwargs))
  if 'trainable' in kwargs and kwargs['trainable'] and 'initial_value' in kwargs:
    name = kwargs['name'] if 'name' in kwargs else 'Anon'
    #print("* Initialized TF variable {}".format(name))
    init_val = kwargs['initial_value']
    del kwargs['initial_value']
    # Mind: init_val could be a callable.
    #print(init_val)
    with tf.device('/gpu:0'):
      if callable(init_val):
        init_val = init_val()
        #print(init_val)
      init_val = _bcast(init_val)
      #print(init_val)
      _tf_Var(self, initial_value=init_val, *args, **kwargs)
  else:
    _tf_Var(self, *args, **kwargs)

# Maybe have function that broadcasts init value for all variables?

# One big AllReduce over all grads.
def _allreduce_n(grads, average=True, device_dense='', device_sparse=''):
    #with tf.device('/gpu:0'):
    return ddl_MDR.all_reduce_n(grads, op='avg' if average else 'sum')

# Reduces gradients in grads_and_vars list.

def _grads_reduce_one(grads_and_vars, average=True):
    # Unzip the list of tuples:
    grads, vars = zip(*grads_and_vars)
    # One big AllReduce over all grads; zip again with the weights:
    return zip(_allreduce_n(grads, average), vars)
    
def _grads_reduce_two(grads_and_vars, average=True):

    divider0 = len(grads_and_vars)/2;
    grads_and_vars0 = grads_and_vars[:divider0]
    grads_and_vars1 = grads_and_vars[divider0:]

    return _grads_reduce_one(grads_and_vars0,average) + _grads_reduce_one(grads_and_vars1,average)
 
dll_group_size = 10000000000
if os.getenv('DDL_GROUP_SIZE') is not None:
    dll_group_size = int(os.getenv('DDL_GROUP_SIZE'))

#Minsik: these two functions are two different styles to group gradients
#both are experimental now, but we use the first one with a huge dll_group_size
#to fallback onto the single allreduce behavior.
#1. group until the collected size is larger than dll_group_size
def _grads_reduce_by_group(grads_and_vars, average=True):
 
    cur_grads_size = 0
    cur_grads_vars = []
    ret_grads_vars = []
    
    for (g, v) in grads_and_vars:
        tensor_size = g.get_shape().num_elements()
        
        #if the current size is too small or adding a new is OK
        if(cur_grads_size+tensor_size < dll_group_size or cur_grads_size < 200000):
            cur_grads_size  +=  tensor_size
            cur_grads_vars.append([g, v])
        #else allreduce
        else:
            if not ret_grads_vars:
                ret_grads_vars += _grads_reduce_one(cur_grads_vars, average)
            else:
                with tf.control_dependencies([ret_grads_vars[-1][0]]):        
                    ret_grads_vars += _grads_reduce_one(cur_grads_vars, average)
                
            #the current one needs to be remembered    
            cur_grads_size = tensor_size
            cur_grads_vars = [(g, v)]
            
    #for any residue        
    if cur_grads_size >0:
        if not ret_grads_vars:
            ret_grads_vars += _grads_reduce_one(cur_grads_vars, average)
        else:
            with tf.control_dependencies([ret_grads_vars[-1][0]]):        
                ret_grads_vars += _grads_reduce_one(cur_grads_vars, average)

    return ret_grads_vars
#2. process any tensor larger than dll_group_size individually 
def _grads_reduce_by_size(grads_and_vars, average=True):
 
    small_grads_size = 0
    small_grads_vars = []
    ret_grads_vars = []
    
    for (g, v) in grads_and_vars:
        tensor_size = g.get_shape().num_elements()
        
        #if the current one is too small, keep adding
        if(tensor_size < dll_group_size) :
            small_grads_vars.append([g, v])
            small_grads_size += tensor_size
        #else do allreduce
        else :
            if not ret_grads_vars:
                ret_grads_vars += _grads_reduce_one([(g,v)], average)
            else:
                with tf.control_dependencies([ret_grads_vars[-1][0]]):        
                    ret_grads_vars += _grads_reduce_one([(g,v)], average)
       
        if small_grads_size > dll_group_size:  
            if not ret_grads_vars:
                ret_grads_vars += _grads_reduce_one(small_grads_vars, average)
            else:
                with tf.control_dependencies([ret_grads_vars[-1][0]]):        
                    ret_grads_vars += _grads_reduce_one(small_grads_vars, average)
                    
            small_grads_size = 0
            small_grads_vars = []
                    
    #for all small ones        
    if small_grads_vars:
        with tf.control_dependencies([ret_grads_vars[-1][0]]):        
            ret_grads_vars += _grads_reduce_one(small_grads_vars, average)

    return ret_grads_vars
   
_grads_reduce = _grads_reduce_by_group
    
# Intercept apply_gradients function:
from tensorflow.python.training import optimizer

_tf_Opt_apply = optimizer.Optimizer.apply_gradients

# Note: cannot create control dependencies on existing nodes!
# Nodes once created are immutable.

# Our replacement:
def new_apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # Sort grads_and_vars on var name:
    grads_and_vars.sort(key=lambda e: e[1].name)
    # Collect all gradients:
    grads = [gv[0] for gv in grads_and_vars]
    # Assume list is non-empty!
    (grad0,var0) = grads_and_vars.pop(0)
    # Add control deps from all gradients to this first AllReduce node:
    new_grads_and_vars = []
    with tf.control_dependencies(grads):
        grad = blc_MDR.all_reduce(grad0, op='avg')
        new_grads_and_vars.append((grad, var0))
    # Insert rest of all_reduce operators:
    for (gradi, vari) in grads_and_vars:
        # Add control dep from previous node to this one:
        with tf.control_dependencies([grad]):
            grad = blc_MDR.all_reduce(gradi, op='avg')
            new_grads_and_vars.append((grad, vari))
    return _tf_Opt_apply(self, new_grads_and_vars, global_step, name)

# Alternative that uses the new all_reduce_n node:
def new_apply_gradients_flat(self, grads_and_vars, global_step=None, name=None):
    grads_and_vars = _grads_reduce(grads_and_vars, average=True)
    return _tf_Opt_apply(self, grads_and_vars, global_step, name)

if mpi_init_status:
    if DDL_OPTIONS is None:
        # User must explicitly call API functions; no override.
        print('DDL: Expect explicit API calls.')
        init = _init
        bcast = _bcast
        allreduce_n = _allreduce_n
        grads_reduce = _grads_reduce
    else:
        # User indicates init should happen at import-time, i.e., now.
        _init(DDL_OPTIONS)

        # Redefine:
        tf.Variable.__init__ = newVar

        # Redefine:
        optimizer.Optimizer.apply_gradients = new_apply_gradients_flat
else:
    def raiseErr(*args):
        raise RuntimeError('Could not initialize DDL. Run this program using mpirun.')

    init = raiseErr
    bcast = raiseErr
    allreduce_n = raiseErr
    grads_reduce = raiseErr
