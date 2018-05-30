import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
import random
import time
import yaml

# TO DO:
# add gradient clipping?

# TO TEST:
# enforcing norm constraint on P
# how to do exploration
# whether this works at all?

# carries out mat*v on all v in batch_v
# mat = [n, m], batch_v = [batch_size, m], returns [batch_size, n]
def batch_matmul(mat, batch_v, name='batch_matmul'):
    with tf.name_scope(name):
        return tf.transpose(tf.matmul(mat,tf.transpose(batch_v)))

def summarize_matrix(name, matrix):
    with tf.name_scope(name):
        eigvals, _ = tf.linalg.eigh(matrix)
        tf.summary.scalar('max_eig', tf.reduce_max(eigvals))
        tf.summary.scalar('min_eig', tf.reduce_min(eigvals))
        tf.summary.scalar('mean_eig', tf.reduce_mean(eigvals))

class klqr:
    # not currently doing value updates at varying rates
    # not currently doing double Q learning (what would this look like?)
    
    def __init__(self,config,sess):
        self.sess = sess
        
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']
        self.a_dim = config['a_dim']
        self.lr = config['lr']
        self.horizon = config['horizon']
        self.gamma = config['discount_rate']

        
        ou_theta = config['ou_theta']
        ou_sigma = config['ou_sigma']
        self.config = config
        
        # Ornstein-Uhlenbeck noise for exploration -- code from Yuke Zhu
        self.noise_var = tf.Variable(tf.zeros([self.a_dim,1]))
        noise_random = tf.random_normal([self.a_dim,1], stddev=ou_sigma)
        self.noise = self.noise_var.assign_sub((ou_theta) * self.noise_var - noise_random)

        self.max_riccati_updates = config['max_riccati_updates']
        self.train_batch_size = config['train_batch_size']
        self.replay_buffer = ReplayBuffer(buffer_size=config['replay_buffer_size'])
        
        self.dynamics_weight = 1.0
        self.cost_weight = 1.0
        self.td_weight = 0.0
        
        self.experience_count = 0
        
        self.updates_so_far = 0
        
    def build_model(self):        

        with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
            
            self.x_ = tf.placeholder(tf.float32,shape=[None, self.x_dim], name='x_')
            self.xp_ = tf.placeholder(tf.float32,shape=[None, self.x_dim], name='xp_')
            self.a_ = tf.placeholder(tf.float32,shape=[None, self.a_dim], name='a_')
            self.r_ = tf.placeholder(tf.float32,shape=[None], name='r_')
            
            self.z = self.encoder(self.x_)
            self.zp = self.encoder(self.xp_)

            print('z shape:', self.z.get_shape())

            #init R

            self.R_asym = tf.get_variable('R_asym',shape=[self.a_dim,self.a_dim])
    #         self.R_asym = tf.Variable(np.random.rand(self.a_dim,self.a_dim) - 0.5)

            # working with Ra.T Ra so that inner product is norm(Rx) and not norm(R.T x)
            self.R = tf.matmul(tf.linalg.transpose(self.R_asym),self.R_asym)

            #init Q -- shape: z_dim * z_dim
            self.Q_asym = tf.get_variable('Q_asym',shape=[self.z_dim,self.z_dim])
            self.Q = tf.matmul(tf.linalg.transpose(self.Q_asym),self.Q_asym)

            #init P -- shape: z_dim * z_dim
            self.P = tf.get_variable('P',shape=[self.z_dim,self.z_dim],trainable=False, initializer=tf.initializers.identity)
            self.P_asym = tf.linalg.transpose(tf.cholesky(self.P))

            #init B -- shape: z_dim * u_dim
            self.B = tf.get_variable('B',shape=[self.z_dim,self.a_dim])
    #         self.B = tf.Variable(np.random.rand(self.z_dim,self.u_dim) - 0.5)

            #init A -- shape: z_dim * z_dim
            self.A = tf.get_variable('A',shape=[self.z_dim,self.z_dim])
    #         self.A = tf.Variable(np.random.rand(self.z_dim,self.z_dim) - 0.5)

            #define K -- shape: u_dim * z_dim
            with tf.name_scope('compute_K'):
                term1 = tf.matrix_inverse(self.R + tf.linalg.transpose(self.B) @ self.P @ self.B)
                term2 = tf.linalg.transpose(self.B) @ self.P @ self.A
                self.K = tf.stop_gradient( -term1 @ term2 )
                self.policy_action = batch_matmul(self.K, self.z)
            
            # predict next state
            with tf.name_scope('predict_next_state'):
                self.zp_pred = batch_matmul(self.A, self.z) + batch_matmul(self.B, self.a_)
            
                        
            #make reward negative to convert to cost
            with tf.name_scope('compute_bootstrapped_v'):
                self.bootstrapped_value = -self.r_ + self.gamma*tf.square(tf.norm(batch_matmul(self.P_asym, self.zp), axis=1))

            with tf.name_scope('compute_action_cost'):
                action_cost = tf.square(tf.norm(batch_matmul(self.R_asym, self.a_), axis=1))#can simplify this by taking norm on other axis

            with tf.name_scope('compute_state_cost'):
                state_cost = tf.square(tf.norm(batch_matmul(self.Q_asym, self.z), axis=1)) 
                
            with tf.name_scope('compute_Qsa'):
                Vzp = tf.square(tf.norm(batch_matmul(self.P_asym, self.zp_pred), axis=1))
                self.Qsa = action_cost + state_cost + Vzp

            with tf.name_scope('predict_reward'):
                self.r_pred = - action_cost - state_cost

            with tf.name_scope('td_loss'):
                self.td_loss = tf.reduce_mean(tf.square(self.bootstrapped_value - self.Qsa))
            with tf.name_scope('dynamics_loss'):
                self.dynamics_loss = tf.reduce_mean(tf.square(self.zp - self.zp_pred))
            with tf.name_scope('cost_loss'):
                self.cost_pred_loss = tf.reduce_mean(tf.square(self.r_pred - self.r_))
            
            
            self.loss = self.td_weight*self.td_loss + self.dynamics_weight*self.dynamics_loss + self.cost_weight*self.cost_pred_loss
            global_step = tf.Variable(0, trainable=False, name='global_step')
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)
            
            # utilities for doing riccati recursion
            self.reset_P_op = self.P.assign(np.zeros([self.z_dim, self.z_dim]))# tf.stop_gradient(self.Q))
            self.riccati_update_op = self.P.assign(tf.stop_gradient(self.riccati_recursion_step()))
            
            # record summaries
            tf.summary.scalar('dynamics_loss', self.dynamics_loss)
            tf.summary.scalar('cost_pred_loss', self.cost_pred_loss)
            tf.summary.scalar('td_loss', self.td_loss)
            summarize_matrix('A', self.A)
#             summarize_matrix('B', self.B)
            summarize_matrix('Q', self.Q)
            summarize_matrix('R', self.R)
            summarize_matrix('P', self.P)
            
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('summaries/'+str(time.time()), self.sess.graph)
            
            self.sess.run(tf.global_variables_initializer())

    
    def update_model(self):        
        #this function is mostly taken from Yuke's code
        if self.replay_buffer.count() < self.train_batch_size:
            return
        
        batch           = self.replay_buffer.getBatch(self.train_batch_size)
        
        states          = np.zeros((self.train_batch_size, self.x_dim))
        rewards         = np.zeros((self.train_batch_size))
        actions         = np.zeros((self.train_batch_size, self.a_dim))
        next_states     = np.zeros((self.train_batch_size, self.x_dim))

        for k, (s0, a, r, s1, done) in enumerate(batch):
            #currently throwing away done states; should fix this
            states[k] = s0
            rewards[k] = r
            actions[k] = a
            next_states[k] = s1
            # check terminal state
#             if not done:
#                 next_states[k] = s1
#                 next_state_mask[k] = 1

        summary, _ = self.sess.run([self.merged, self.train_op],
        {
        self.x_:  states,
        self.xp_: next_states,
        self.a_:  actions,
        self.r_:  rewards
        })
    
        self.train_writer.add_summary(summary, self.updates_so_far)
        self.updates_so_far += 1
    
        #possibly update target via Riccati recursion? or do standard target separation? 
    
    def riccati_recursion_step(self):
        with tf.name_scope('riccati_step'):
    #         ABK = self.A + self.B @ self.K
    #         APA = tf.transpose(ABK) @ self.P @ ABK 
    #         return self.Q + tf.transpose(self.K) @ self.R @ self.K + self.gamma*APA
            newP =  self.Q + tf.transpose(self.A) @ self.P @ self.A - tf.transpose(self.A) @ self.P @ self.B @ tf.matrix_inverse(self.R + tf.transpose(self.B) @ self.P @ self.B ) @ tf.transpose(self.B) @ tf.transpose(self.P) @ self.A
            return 0.5*(newP + tf.transpose(newP))
        
    def update_P(self):
#         print('updating P')
#         reset_q_op = 
#         self.P = tf.identity(self.Q)
#         for k in range(self.max_riccati_updates):
#             #do Riccati backup in tensorflow oh god why
#             ABK = self.A + tf.matmul(self.B,self.K)
#             APA = tf.matmul(tf.matmul(tf.transpose(ABK),self.P),ABK) #
#             self.P = self.Q + tf.matmul(tf.matmul(tf.transpose(self.K),self.R),self.K) + self.gamma*APA
        
#         self.P_asym = tf.transpose(tf.cholesky(self.P))
        self.sess.run(self.reset_P_op)
        for k in range(self.max_riccati_updates):
            self.sess.run(self.riccati_update_op)
        

        print(self.sess.run(self.P))
            #TODO add a termination criterion for norm of Riccati update difference?
    
    def pi(self,x,explore=True):
        self.experience_count += 1
        x = np.reshape(x,(1,3))
        
        a,w = self.sess.run([self.policy_action,self.noise], {self.x_: x})
        
        a = a + w if explore else a
        # TODO check the dimension of the output of this
        return [a[0,0]]
        
    def store_experience(self,s,a,r,sp,done):
        # currently storing experience for every iteration
        self.replay_buffer.add(s, a, r, sp, done)
    
    def encoder(self,x,name="encoder",batch_norm=True):
        layer_sizes = self.config['encoder_layers']
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            inp = x
            for units in layer_sizes: 
                inp = tf.layers.dense(inputs=inp, units=units,activation=tf.nn.relu)

            z = tf.layers.dense(inputs=inp, units=self.z_dim,activation=None)

        if batch_norm:
            z = tf.layers.batch_normalization(z)

        return z

class ReplayBuffer:
    # taken from Yuke Zhu's Q learning implementation
    
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # random draw N
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, next_action, done):
        new_experience = (state, action, reward, next_action, done)
        if self.num_experiences < self.buffer_size:
          self.buffer.append(new_experience)
          self.num_experiences += 1
        else:
          self.buffer.popleft()
          self.buffer.append(new_experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0