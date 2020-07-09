import numpy as np 
import tensorflow as tf
import time
import numpy.linalg as alg
from util.poseAugment import PoseAugment
import os
import cv2
    
class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.image_c_dim = args.image_c_dim
        self.pose_dim = args.pose_dim
        self.num_jnt = args.num_jnt 
        self.num_fea = args.num_fea
        self.args = args
        if args.phase == 'train':
            self.is_training = True
            self.tf_is_training = tf.constant(True, dtype=tf.bool)
        else:
            self.is_training = False
            self.tf_is_training = tf.constant(False, dtype=tf.bool)
            
        self.poseAugmentor = PoseAugment(args.dataset)
        self.aug_modes = ['off', 'rot', 'sc', 'none']

        self.weight_decay = 0.5*1e-4
        self.regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
        self.HUBER_DELTA = 0.01
        self.num_stack = 2
        
        if args.dataset == 'NYU':
            A = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1]])
        elif args.dataset == 'ICVL':
            A = np.array([[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
        elif args.dataset == 'MSRA':
            A = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    
        D = np.diag(np.power(np.sum(A, axis=0), -0.5))
        a = np.dot(np.dot(D, A), D)
        a = np.expand_dims(a, 0).astype(dtype=np.float32)
        self.A = tf.constant(a, dtype=tf.float32)
        self.B = tf.Variable(tf.constant(1e-6, dtype=tf.float32, shape=[1, self.num_jnt, self.num_jnt]), name='edge')
                  
        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=10)
        
    def _build_model(self):
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.num_jnt], name='z')
        self.image = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size,
                                                         self.image_c_dim], name='image')
        self.uv = tf.placeholder(tf.float32, [self.batch_size, self.num_jnt, 2], name='uv')
        self.u, self.v = tf.split(self.uv, 2, -1)
        self.u = tf.squeeze(self.u, [-1])
        self.v = tf.squeeze(self.v, [-1])
        self.cube = tf.placeholder(tf.float32, [self.batch_size], name='cube')
        self.is_training_pholder = tf.placeholder(tf.bool, name='is_training')
        
        self.end_points = {}
        self.end_points['mapping_weights'] = []
        self.end_points['hm_u_outs'] = []
        self.end_points['hm_v_outs'] = []
        self.end_points['hm_z_outs'] = []

        self.input_size = self.image_size
        
        # 96*96 -> 48*48
        self.conv_1 = self.conv_bn_relu(self.image, 32, kernel_size=7, strides=2)
        self.conv_2 = self.residual(self.conv_1, 64) 
        # 48*48 -> 24*24
        self.pool_1 = tf.nn.max_pool(self.conv_2, [1,2,2,1], [1,2,2,1], padding='SAME')
        self.conv_3 = self.residual(self.pool_1)
        self.conv_4 = self.residual(self.conv_3, self.num_fea)
        hg_ins = self.conv_4
        
        self.output_size = int(self.input_size/4) 
        
        for i in range(self.num_stack):
            hg_outs = self.hourglass(hg_ins, n=3)
            
            ll = self.residual(hg_outs)
            
            # graph reasoning
            graph_reasoning_feats, mapping_weights = self.graph_reasoning(ll)
            
            self.end_points['mapping_weights'].append(mapping_weights)
            
            ll_1 = self.conv_bn_relu(tf.concat([ll, graph_reasoning_feats], axis=-1), self.num_fea, kernel_size=1, strides=1)
            
            # hm_outputs
            hm_u_out = self.conv2d(ll_1, self.num_jnt, kernel_size=1, strides=1, use_bias=False)
            self.end_points['hm_u_outs'].append(hm_u_out)
            hm_v_out = self.conv2d(ll_1, self.num_jnt, kernel_size=1, strides=1, use_bias=False)
            self.end_points['hm_v_outs'].append(hm_v_out)            
            hm_z_out = self.conv2d(ll_1, self.num_jnt, kernel_size=1, strides=1, use_bias=False)
            self.end_points['hm_z_outs'].append(hm_z_out)
            
                       
            if i < self.num_stack-1:
                tmp_out = tf.concat([hm_u_out, hm_v_out, hm_z_out], axis=-1)   
                tmp_out_reshaped = self.conv_bn_relu(tmp_out, self.num_fea, kernel_size=1, strides=1)                
                inter = self.conv_bn_relu(ll_1, self.num_fea, kernel_size=1, strides=1)             
                hg_ins = hg_ins + tmp_out_reshaped + inter 
                                                                            
        x = list(range(0, self.output_size))
        y = list(range(0, self.output_size))
        X, Y = tf.meshgrid(x,y)
        X = tf.to_float(tf.expand_dims(X, -1))
        Y = tf.to_float(tf.expand_dims(Y, -1))   
        
        self.u_rs = tf.expand_dims(tf.expand_dims(self.u, -2), -2)
        self.u_rs = tf.tile(self.u_rs, [1, self.output_size, self.output_size, 1])
        self.hm_u = self.u_rs - X/(self.output_size-1)
        self.v_rs = tf.expand_dims(tf.expand_dims(self.v, -2), -2)
        self.v_rs = tf.tile(self.v_rs, [1, self.output_size, self.output_size, 1])
        self.hm_v = self.v_rs - Y/(self.output_size-1)
        self.z_rs = tf.expand_dims(tf.expand_dims(self.z, -2), -2)
        self.z_rs = tf.tile(self.z_rs, [1, self.output_size, self.output_size, 1])
        self.hm_z = self.z_rs - tf.image.resize_images(self.image, (self.output_size, self.output_size), tf.image.ResizeMethod.BILINEAR)
        
        self.loss_uv_list = []
        self.loss_z_list = []
        self.loss_hm_u_list = []
        self.loss_hm_v_list = []
        self.loss_hm_z_list = []
        
        for i in range(self.num_stack):           
            u = tf.reduce_sum(self.end_points['mapping_weights'][i] * (self.end_points['hm_u_outs'][i] + X/(self.output_size-1)), [1,2])
            v = tf.reduce_sum(self.end_points['mapping_weights'][i] * (self.end_points['hm_v_outs'][i] + Y/(self.output_size-1)), [1,2])
            uv = tf.stack([u, v], -1) 
            loss_uv = self.smoothL1(tf.reshape(self.uv, [-1]), tf.reshape(uv, [-1]))        
            self.loss_uv_list.append(loss_uv)
            z = tf.reduce_sum(self.end_points['mapping_weights'][i] * (self.end_points['hm_z_outs'][i] + 
                             tf.image.resize_images(self.image, (self.output_size, self.output_size), 
                                                    tf.image.ResizeMethod.BILINEAR)), [1,2])
            loss_z = self.smoothL1(tf.reshape(self.z, [-1]), tf.reshape(z, [-1]))
            self.loss_z_list.append(loss_z)
            
            loss_hm_u = self.smoothL1(tf.reshape(self.hm_u, [-1]), tf.reshape(self.end_points['hm_u_outs'][i], [-1])) * 0.0001
            self.loss_hm_u_list.append(loss_hm_u)
            loss_hm_v = self.smoothL1(tf.reshape(self.hm_v, [-1]), tf.reshape(self.end_points['hm_v_outs'][i], [-1])) * 0.0001
            self.loss_hm_v_list.append(loss_hm_v)
            loss_hm_z = self.smoothL1(tf.reshape(self.hm_z, [-1]), tf.reshape(self.end_points['hm_z_outs'][i], [-1])) * 0.0001
            self.loss_hm_z_list.append(loss_hm_z)
            
        self.output_z = z * tf.expand_dims(self.cube, 1) / 2
        self.output_uv = uv * self.image_size
        
        self.loss_uv, self.loss_z = tf.add_n(self.loss_uv_list), tf.add_n(self.loss_z_list)
        self.loss_hm_u, self.loss_hm_v, self.loss_hm_z = tf.add_n(self.loss_hm_u_list), tf.add_n(self.loss_hm_v_list), tf.add_n(self.loss_hm_z_list)        
        self.loss_net = self.loss_uv + self.loss_z + self.loss_hm_u + self.loss_hm_v + self.loss_hm_z
        reg_set=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        for key in reg_set:
#            print(key.name)
        self.loss_reg = tf.add_n(reg_set)
        self.loss = self.loss_net + self.loss_reg 
        
#        self.loss_sum = tf.summary.scalar("loss", self.loss)                 
        
        self.t_vars = tf.trainable_variables()
#        for var in self.t_vars: print(var.name)

        # statistics of model parameters
        trainable_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print (variable.name, shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            trainable_parameters += variable_parameters
         
        print('number of trainable parameters %d'%(trainable_parameters))
    
    def augmentPose(self, dpt, com2D, joints3D, M, aug_modes, cube, seed, 
                    sigma_off=None, sigma_sc=None, rot_range=None):
        """
        Commonly used function to augment hand poses
        :param dpt: cropped depth image
        :param com2D: center of mass in image coordinates (x,y,z)
        :param joints3D: 3D annotation, related to the com3D
        :param M: affine translation matrix from orignal image coordinate to cropped image coordiante
        :param aug_modes: augmentation modes
        :param cube: cube
        :param sigma_off: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: depth image, 3D annotations, 2D annotations
        """
        assert cube is not None
        
        assert len(dpt.shape) == 2
        assert isinstance(aug_modes, list)
        
        if sigma_off is None:
            sigma_off = 10.

        if sigma_sc is None:
            sigma_sc = 0.05

        if rot_range is None:
            rot_range = 180.
           
        rng = np.random.RandomState(seed)
        mode = rng.randint(0, len(aug_modes))
        off = rng.randn(3) * sigma_off  # +-px/mm
        off = np.clip(off, -10, 10)
        rot = rng.uniform(-rot_range, rot_range)
        sc = abs(1. + rng.randn() * sigma_sc)
        sc = np.clip(sc, 0.9, 1.1)
        
        if aug_modes[mode] == 'off':
            new_img, new_joints3D, new_joints2D, new_com2D, Mnew, new_com3D = self.poseAugmentor.moveCoM(dpt, com2D, off, joints3D, M, cube)
        elif aug_modes[mode] == 'rot':
            new_img, new_joints3D, new_joints2D, new_com2D, Mnew, new_com3D = self.poseAugmentor.rotateHand(dpt, com2D, rot, joints3D, M, cube)
        elif aug_modes[mode] == 'sc':
            new_img, new_joints3D, new_joints2D, new_com2D, Mnew, new_com3D = self.poseAugmentor.scaleHand(dpt, com2D, sc, joints3D, M, cube)
        elif aug_modes[mode] == 'none':
            new_img, new_joints3D, new_joints2D, new_com2D, Mnew, new_com3D = self.poseAugmentor.scaleHand(dpt, com2D, 1, joints3D, M, cube)
        else:
            raise NotImplementedError()
                   
        return new_img, new_joints3D, new_joints2D, aug_modes[mode]
            
    def train(self, args, train_stream, val_stream=None, extra_test_data=None):
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        update_os = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_os):
            self.optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.loss, var_list=self.t_vars)
            
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
#        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
#        print(self.sess.run(self.t_vars[1]))
        counter = 0
        last_e = 100
        start_time = time.time()
        
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
        for epoch in range(args.epoch): 
            lr = args.lr * 0.96 ** epoch
            
            for data in train_stream.iterate(batchsize = self.batch_size, seed = epoch):
                batch_image, batch_pose, batch_M, batch_com2D, batch_cube = data
                
                # Pose augmentation
                batch_image_aug = batch_image.copy()
                batch_pose_aug = batch_pose.copy()
                batch_uv_aug = np.zeros((batch_pose.shape[0], batch_pose.shape[1], 2), np.float32)
                batch_z_aug = np.zeros((batch_pose.shape[0], batch_pose.shape[1]), np.float32)    
                
                for k in range(self.batch_size):
                    imgD, batch_pose_aug[k], batch_uv_aug[k], aug_mode = self.augmentPose(batch_image[k], 
                                                        batch_com2D[k], batch_pose[k], batch_M[k], self.aug_modes, batch_cube[k], counter+k)
                    batch_pose_aug[k] /= (batch_cube[k][0] / 2)
                    batch_pose_aug[k] = np.clip(batch_pose_aug[k], -1, 1)
                    
                    premax = imgD.max()
                    imgD[imgD == premax] = batch_com2D[k][2] + (batch_cube[k][2] / 2.)
                    imgD[imgD == 0] = batch_com2D[k][2] + (batch_cube[k][2] / 2.)
                    imgD[imgD >= batch_com2D[k][2] + (batch_cube[k][2] / 2.)] = batch_com2D[k][2] + (batch_cube[k][2] / 2.)
                    imgD[imgD <= batch_com2D[k][2] - (batch_cube[k][2] / 2.)] = batch_com2D[k][2] - (batch_cube[k][2] / 2.)
                    imgD -= batch_com2D[k][2]
                    imgD /= (batch_cube[k][2] / 2.)
                    batch_image_aug[k] = imgD
                    
                batch_image_aug = np.reshape(batch_image_aug, (batch_image.shape[0], batch_image.shape[1], batch_image.shape[2], 1))
                batch_uv_aug /= (self.image_size-1)
                batch_uv_aug = np.clip(batch_uv_aug, -1, 1)
                batch_z_aug = batch_pose_aug[:, :, 2]
                
                self.current_batch_size = batch_image_aug.shape[0]
                
                # Update network
                self.warmup_iters = 100
                self.warmup_ratio = 1/3
                if epoch == 0 and counter < self.warmup_iters:
                    k = (1 - (counter + 1) / self.warmup_iters) * (1 - self.warmup_ratio)
                    lr = (1 - k) * args.lr
                _ = self.sess.run(self.optim,
                               feed_dict={self.image: batch_image_aug,
                               self.uv: batch_uv_aug, self.z: batch_z_aug, self.cube: batch_cube[:, 0],
                               self.lr: lr, self.is_training_pholder: self.is_training})
   
                # self.writer.add_summary(summary_str, counter) 
                                                               
                counter += 1
                if np.mod(counter, 100) == 1:
                    loss, loss_start_uv, loss_end_uv, loss_start_z, loss_end_z, loss_reg, \
                    loss_start_hm_u, loss_end_hm_u, loss_start_hm_v, loss_end_hm_v, loss_start_hm_z, loss_end_hm_z = self.sess.run(
                    [self.loss, self.loss_uv_list[0], self.loss_uv_list[-1], self.loss_z_list[0], self.loss_z_list[-1], 
                     self.loss_reg, self.loss_hm_u_list[0], self.loss_hm_u_list[-1],
                     self.loss_hm_v_list[0], self.loss_hm_v_list[-1], self.loss_hm_z_list[0], self.loss_hm_z_list[-1]],                      
                    feed_dict={self.image: batch_image_aug,
                               self.uv: batch_uv_aug, self.z: batch_z_aug, self.cube: batch_cube[:, 0],
                               self.is_training_pholder: self.is_training})
                    print("Epoch: [%2d] [%4d] lr: [%1.6f] loss: total-[%4.4f] s_uv-[%4.4f] e_uv-[%4.4f] s_z-[%4.4f] e_z-[%4.4f] reg-[%4.4f] s_hm_u-[%4.4f] e_hm_u-[%4.4f] s_hm_v-[%4.4f] e_hm_v-[%4.4f] s_hm_z-[%4.4f] e_hm_z-[%4.4f] aug_mode-[%s] time: %4.4f" % (
                        epoch+1, counter, lr, loss, loss_start_uv, loss_end_uv, loss_start_z, loss_end_z, 
                        loss_reg, loss_start_hm_u, loss_end_hm_u, loss_start_hm_v, loss_end_hm_v, loss_start_hm_z, loss_end_hm_z,
                        aug_mode, time.time()-start_time))  
            
            #evaluation
            if not val_stream is None:
                import collections
                CameraConfig = collections.namedtuple('CameraConfig', 'fx,fy,cx,cy,w,h')
                if args.dataset == 'NYU':
                    cfg = CameraConfig(fx=588.235, fy=587.084, cx=320, cy=240, w=640, h=480)
                else:
                    cfg = CameraConfig(fx=241.42, fy=241.42, cx=160, cy=120, w=320, h=240)
                
                def transformPoint2D(pt, M):
                    """
                    Transform point in 2D coordinates
                    :param pt: point coordinates
                    :param M: transformation matrix
                    :return: transformed point
                    """
                    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
                    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])
                
                def transformPoints2D(pts, M):
                    """
                    Transform points in 2D coordinates
                    :param pts: point coordinates
                    :param M: transformation matrix
                    :return: transformed points
                    """
                    ret = pts.copy()
                    for i in range(pts.shape[0]):
                        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
                    return ret
                
                xyz_error = []
                for data in val_stream.iterate(batchsize = self.batch_size, seed = 0, shuffle=False):
                    batch_image, batch_pose, batch_M, batch_com3D, batch_uv, batch_cube = data
                    batch_image = np.reshape(batch_image, (batch_image.shape[0], batch_image.shape[1], batch_image.shape[2], 1))
                    self.current_batch_size =  batch_image.shape[0]          
                    inference_uv, inference_z = self.sess.run([self.output_uv, self.output_z],
                          feed_dict={self.image: batch_image, self.cube: batch_cube[:, 0],
                          self.is_training_pholder: False})
    
                    inference_uv_orig = inference_uv.copy()
                    for i in range(self.batch_size):
                        inference_uv_orig[i] = transformPoints2D(inference_uv[i], np.linalg.inv(batch_M[i]))
                    inference_uv = inference_uv_orig
                    
                    inference_pose_backpro = batch_pose.copy()
                    inference_pose_backpro[:,:,2] = inference_z + np.expand_dims(batch_com3D[:, 2], -1)
                    inference_pose_backpro[:,:,0] = np.multiply(inference_uv[:,:,0]-cfg[2], inference_pose_backpro[:,:,2])/cfg[0] 
                    inference_pose_backpro[:,:,1] = np.multiply(cfg[3]-inference_uv[:,:,1], inference_pose_backpro[:,:,2])/cfg[1]
                    
                    
                    inference_pose = inference_pose_backpro - batch_com3D[:, np.newaxis, :]
                    
                    for i in range(self.batch_size):
                        xyz_error.append(self.meanJntError(batch_pose[i], inference_pose[i]))
                
                # test extra data        
                if not extra_test_data is None:
                    batch_image, batch_pose, batch_M = extra_test_data[0], extra_test_data[1], extra_test_data[2] 
                    batch_com3D, batch_uv, batch_cube = extra_test_data[3], extra_test_data[4], extra_test_data[5]
                    batch_image = np.reshape(batch_image, (batch_image.shape[0], batch_image.shape[1], batch_image.shape[2], 1))
                    self.current_batch_size =  batch_image.shape[0]          
                    inference_uv, inference_z = self.sess.run([self.output_uv, self.output_z],
                          feed_dict={self.image: batch_image, self.cube: batch_cube[:, 0],
                          self.is_training_pholder: False})
    
                    inference_uv_orig = inference_uv.copy()
                    for i in range(batch_image.shape[0]):
                        inference_uv_orig[i] = transformPoints2D(inference_uv[i], np.linalg.inv(batch_M[i]))
                    inference_uv = inference_uv_orig
                    
                    inference_pose_backpro = batch_pose.copy()
                    inference_pose_backpro[:,:,2] = inference_z + np.expand_dims(batch_com3D[:, 2], -1)
                    inference_pose_backpro[:,:,0] = np.multiply(inference_uv[:,:,0]-cfg[2], inference_pose_backpro[:,:,2])/cfg[0] 
                    inference_pose_backpro[:,:,1] = np.multiply(cfg[3]-inference_uv[:,:,1], inference_pose_backpro[:,:,2])/cfg[1]
                    
                    
                    inference_pose = inference_pose_backpro - batch_com3D[:, np.newaxis, :]
                    
                    for i in range(batch_image.shape[0]):
                        xyz_error.append(self.meanJntError(batch_pose[i], inference_pose[i]))
                
                meane = np.mean(xyz_error)
                print ('mean xyz error: %f'%(meane))
                
                if args.dataset == 'MSRA':
                    logt = open(args.log_dir + '/log_dense_offset_graph_A_test_sub_' + str(args.test_sub) + '.txt', 'a+')
                else:
                    logt = open(args.log_dir + '/log_dense_offset_graph_A.txt', 'a+')
                logt.write('epoch {}, mean error {}'.format(epoch+1, meane))
                logt.write('\n')
                logt.close()
        
                if last_e>=meane:
                    last_e=meane
                    if args.dataset == 'MSRA':
                        logt = open(args.log_dir + '/log_dense_offset_graph_A_test_sub_' + str(args.test_sub) + '.txt', 'a+')
                    else:
                        logt = open(args.log_dir + '/log_dense_offset_graph_A.txt', 'a+')
                    logt.write("*********************")
                    logt.write('\n')
                    logt.write('current best epoch is {}, mean error is {}'.format(epoch+1, last_e))
                    logt.write('\n')
                    logt.write("*********************")
                    logt.write('\n')
                    logt.close()
            
            if args.dataset == 'MSRA':
                self.save(args.checkpoint_dir + '/test_sub_' + str(args.test_sub), epoch+1)
            else:
                self.save(args.checkpoint_dir, epoch+1)
                          
        print('Training finished. Saving final snapshot.')
        if args.dataset == 'MSRA':
            self.save(args.checkpoint_dir + '/test_sub_' + str(args.test_sub), counter)
        else:
            self.save(args.checkpoint_dir, counter)

    def test(self, args, val_stream, extra_test_data):
        if args.dataset == 'MSRA':
          checkpoint_path = args.checkpoint_dir + '/test_sub_' + str(args.test_sub)
        else:
          checkpoint_path = args.checkpoint_dir
        if self.load(checkpoint_path):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            os._exit(0)
        
        count = 0

        xyz_error = []
        uv_error = []
        xy_error = []
        z_error = []
        max_error = []
        joints_error = []
        
        import collections
        CameraConfig = collections.namedtuple('CameraConfig', 'fx,fy,cx,cy,w,h')
        if args.dataset == 'NYU':
            cfg = CameraConfig(fx=588.235, fy=587.084, cx=320, cy=240, w=640, h=480)
        else:
            cfg = CameraConfig(fx=241.42, fy=241.42, cx=160, cy=120, w=320, h=240)
            
        f = open('results/'+args.dataset+'_test_error_list.txt', 'a')
        f1 = open('results/'+args.dataset+'_JGR-P2O.txt', 'a')
        
        def transformPoint2D(pt, M):
            """
            Transform a point in 2D coordinates
            :param pt: point coordinates
            :param M: transformation matrix
            :return: transformed point
            """
            pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
            return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])
            
        def transformPoints2D(pts, M):
            """
            Transform points in 2D coordinates
            :param pts: point coordinates
            :param M: transformation matrix
            :return: transformed points
            """
            ret = pts.copy()
            for i in range(pts.shape[0]):
                ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
            return ret
        
        # test test_stream
        for data in val_stream.iterate(batchsize = self.batch_size, seed = 0, shuffle=False):
            batch_image, batch_pose, batch_M, batch_com3D, batch_uv, batch_cube = data
            batch_image = np.reshape(batch_image, (batch_image.shape[0], batch_image.shape[1], batch_image.shape[2], 1))        
            self.current_batch_size =  batch_image.shape[0]           
            inference_uv, inference_z = self.sess.run([self.output_uv, self.output_z],
                      feed_dict={self.image: batch_image, self.cube: batch_cube[:, 0],
                      self.is_training_pholder: False})

            inference_uv_orig = inference_uv.copy()
            for i in range(self.batch_size):
                inference_uv_orig[i] = transformPoints2D(inference_uv[i], np.linalg.inv(batch_M[i]))
            inference_uv = inference_uv_orig
            
            inference_pose_backpro = batch_pose.copy()
            inference_pose_backpro[:,:,2] = inference_z + np.expand_dims(batch_com3D[:, 2], -1)
            inference_pose_backpro[:,:,0] = np.multiply(inference_uv[:,:,0]-cfg[2], inference_pose_backpro[:,:,2])/cfg[0] 
            inference_pose_backpro[:,:,1] = np.multiply(cfg[3]-inference_uv[:,:,1], inference_pose_backpro[:,:,2])/cfg[1]
                        
            inference_pose = inference_pose_backpro - batch_com3D[:, np.newaxis, :]

            batch_xyz_error = []
            batch_uv_error = []
            batch_xy_error = []
            batch_z_error = []
            for i in range(self.batch_size):
                xyz_error.append(self.meanJntError(batch_pose[i], inference_pose[i]))
                f.write(str(xyz_error[-1]))
                f.write(' ')
                for j in range(self.num_jnt):
                  f1.write(str(inference_uv_orig[i][j][0]))
                  f1.write(' ')
                  f1.write(str(inference_uv_orig[i][j][1]))
                  f1.write(' ')
                  f1.write(str(inference_pose_backpro[i][j][-1]))
                  f1.write(' ')
                f1.write('\n')
                batch_xyz_error.append(self.meanJntError(batch_pose[i], inference_pose[i]))
                xy_error.append(self.meanJntError(batch_pose[i][:, :-1], inference_pose[i][:, :-1]))
                batch_xy_error.append(self.meanJntError(batch_pose[i][:, :-1], inference_pose[i][:, :-1]))
                z_error.append(np.abs(batch_pose[i][:, -1]-inference_pose[i][:, -1]))
                batch_z_error.append(np.abs(batch_pose[i][:, -1]-inference_pose[i][:, -1]))
                uv_error.append(self.meanJntError(inference_uv[i], batch_uv[i]))
                batch_uv_error.append(self.meanJntError(inference_uv[i], batch_uv[i]))
                max_error.append(self.maxJntError(batch_pose[i], inference_pose[i]))                
                joints_error.append(alg.norm(batch_pose[i]-inference_pose[i], axis=1))

            count += 1
#            print ('batch: %d mean xyz error: %f, mean uv error: %f, mean xy error: %f, mean z error: %f'%(count, 
#                np.mean(batch_xyz_error), np.mean(batch_uv_error), np.mean(batch_xy_error), np.mean(batch_z_error)))
        
        # test extra data
        if extra_test_data is not None:
            batch_image, batch_pose, batch_M = extra_test_data[0], extra_test_data[1], extra_test_data[2] 
            batch_com3D, batch_uv, batch_cube = extra_test_data[3], extra_test_data[4], extra_test_data[5]
            batch_image = np.reshape(batch_image, (batch_image.shape[0], batch_image.shape[1], batch_image.shape[2], 1))       
            self.current_batch_size =  batch_image.shape[0]        
            inference_uv, inference_z = self.sess.run([self.output_uv, self.output_z],
                feed_dict={self.image: batch_image, self.cube: batch_cube[:, 0],
                self.is_training_pholder: False})
            inference_uv_orig = inference_uv.copy()
            for i in range(batch_image.shape[0]):
                inference_uv_orig[i] = transformPoints2D(inference_uv[i], np.linalg.inv(batch_M[i]))
            inference_uv = inference_uv_orig        
            inference_pose_backpro = batch_pose.copy()
            inference_pose_backpro[:,:,2] = inference_z + np.expand_dims(batch_com3D[:, 2], -1)
            inference_pose_backpro[:,:,0] = np.multiply(inference_uv[:,:,0]-cfg[2], inference_pose_backpro[:,:,2])/cfg[0] 
            inference_pose_backpro[:,:,1] = np.multiply(cfg[3]-inference_uv[:,:,1], inference_pose_backpro[:,:,2])/cfg[1]                    
            inference_pose = inference_pose_backpro - batch_com3D[:, np.newaxis, :]
    
            batch_xyz_error = []
            batch_uv_error = []
            batch_xy_error = []
            batch_z_error = []
            for i in range(batch_image.shape[0]):
                xyz_error.append(self.meanJntError(batch_pose[i], inference_pose[i]))
                f.write(str(xyz_error[-1]))
                f.write(' ')
                for j in range(self.num_jnt):
                  f1.write(str(inference_uv_orig[i][j][0]))
                  f1.write(' ')
                  f1.write(str(inference_uv_orig[i][j][1]))
                  f1.write(' ')
                  f1.write(str(inference_pose_backpro[i][j][-1]))
                  f1.write(' ')
                f1.write('\n')
                batch_xyz_error.append(self.meanJntError(batch_pose[i], inference_pose[i]))
                xy_error.append(self.meanJntError(batch_pose[i][:, :-1], inference_pose[i][:, :-1]))
                batch_xy_error.append(self.meanJntError(batch_pose[i][:, :-1], inference_pose[i][:, :-1]))
                z_error.append(np.abs(batch_pose[i][:, -1]-inference_pose[i][:, -1]))
                batch_z_error.append(np.abs(batch_pose[i][:, -1]-inference_pose[i][:, -1]))
                uv_error.append(self.meanJntError(inference_uv[i], batch_uv[i]))
                batch_uv_error.append(self.meanJntError(inference_uv[i], batch_uv[i]))
                max_error.append(self.maxJntError(batch_pose[i], inference_pose[i]))                
                joints_error.append(alg.norm(batch_pose[i]-inference_pose[i], axis=1))
#            print ('batch: %d mean xyz error: %f, mean uv error: %f, mean xy error: %f, mean z error: %f'%(count, 
#                np.mean(batch_xyz_error), np.mean(batch_uv_error), np.mean(batch_xy_error), np.mean(batch_z_error)))
     
        f.close()
        f1.close()
        print ('mean xyz error: %f'%(np.mean(xyz_error)))
        print ('mean uv error: %f'%(np.mean(uv_error)))
        print ('mean xy error: %f'%(np.mean(xy_error)))
        print ('mean z error: %f'%(np.mean(z_error)))
        
        joints_error = np.array(joints_error)
        print('Total %d samples have been tested!'%(joints_error.shape[0]))
        print('per joint mean error:')
        print (np.mean(joints_error, 0))
        
        self.calMaxError(max_error)
        
    def maxJntError(self, skel1, skel2):
        diff = skel1 - skel2
        diff = alg.norm(diff, axis=1)
        return diff.max()
    
    def meanJntError(self, skel1, skel2):
        diff = skel1 - skel2
        diff = alg.norm(diff, axis=1)
        return diff.mean()
        
    def calMaxError(self, score_list):
        score_list = sorted(score_list)

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=10.5):
                th_idx += 1
        print ('10mm percentage: %f'%(float(th_idx)/len(score_list)))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=20.5):
                th_idx += 1
        print ('20mm percentage: %f'%(float(th_idx)/len(score_list)))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=30.5):
                th_idx += 1
        print ('30mm percentage: %f'%(float(th_idx)/len(score_list)))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=40.5):
                th_idx += 1
        print ('40mm percentage: %f'%(float(th_idx)/len(score_list)))
                
    def save(self, checkpoint_dir, step):
        model_name = "3D_Detection.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False 
            
#        if ckpt:
#            self.saver.restore(self.sess, os.path.join(checkpoint_dir, "3D_Detection.model-90880"))
#            return True
#        else:
#            return False
    
    def GCN(self, node_feats):
        # node_feats: [batch_size, num_node, num_feats]
        
        node_feats_reshaped = tf.reshape(node_feats, [-1, self.num_fea])
        node_feats_trans = tf.layers.dense(node_feats_reshaped, self.num_fea,
            kernel_initializer=tf.truncated_normal_initializer(0, 0.01), kernel_regularizer=self.regularizer, use_bias=False)
        node_feats_trans_reshaped = tf.reshape(node_feats_trans, [-1, self.num_jnt, self.num_fea])
        
        # skeleton graph       
        node_prop_feats = tf.nn.relu(tf.matmul(tf.tile(self.A, [self.batch_size, 1, 1]), node_feats_trans_reshaped))
        
        # random graph
        # node_prop_feats = tf.nn.relu(tf.matmul(tf.tile(self.B, [self.batch_size, 1, 1]), node_feats_trans_reshaped))
        
        # learned graph
#        theta = tf.layers.dense(node_feats_reshaped, self.num_fea,
#            kernel_initializer=tf.truncated_normal_initializer(0, 0.01), kernel_regularizer=self.regularizer, use_bias=False)
#        theta_reshaped = tf.reshape(theta, [-1, self.num_jnt, self.num_fea])
#        fi = tf.layers.dense(node_feats_reshaped, self.num_fea,
#            kernel_initializer=tf.truncated_normal_initializer(0, 0.01), kernel_regularizer=self.regularizer, use_bias=False)
#        fi_reshaped = tf.reshape(fi, [-1, self.num_jnt, self.num_fea])
#        C = tf.matmul(theta_reshaped, tf.transpose(fi_reshaped, [0, 2, 1]))
#        C = tf.nn.softmax(C, -1)
#        node_prop_feats = tf.nn.relu(tf.matmul(C, node_feats_trans_reshaped))
        
        return node_prop_feats
    
    def graph_reasoning(self, ll):
        # pixel-to-joint voting
        mapping_weights = self.conv2d(ll, self.num_jnt, kernel_size=1, strides=1, use_bias=True)
        mapping_weights_reshaped = tf.reshape(mapping_weights, [-1, self.output_size*self.output_size, self.num_jnt])
        mapping_weights_reshaped = tf.nn.softmax(mapping_weights_reshaped, 1)
        mapping_weights = tf.reshape(mapping_weights_reshaped, [-1, self.output_size, self.output_size, self.num_jnt])
        output_mapping_weights = mapping_weights
        local_feats = self.conv_bn_relu(ll, self.num_fea, kernel_size=1, strides=1)
        local_feats_reshaped = tf.reshape(local_feats, [-1, self.output_size*self.output_size, self.num_fea])
        mapping_weights_reshaped = tf.transpose(mapping_weights_reshaped, perm=[0, 2, 1])          
        node_feats = tf.matmul(mapping_weights_reshaped, local_feats_reshaped)
        
        # graph reasoning
        node_prop_feats = self.GCN(node_feats)
        
        # joint-to-pixel mapping
        node_prop_feats_reshaped = tf.expand_dims(tf.expand_dims(node_prop_feats, -2), -2)
        node_prop_feats_reshaped = tf.tile(node_prop_feats_reshaped, [1, 1, self.output_size, self.output_size, 1])      
        mapping_weights_inverse = tf.expand_dims(tf.transpose(mapping_weights, [0, 3, 1, 2]), -1)
        new_feats = node_prop_feats_reshaped * mapping_weights_inverse           
        new_feats = tf.reduce_mean(new_feats, 1) 
        new_feats = self.conv_bn_relu(new_feats, self.num_fea, kernel_size=1, strides=1)
       
        return new_feats, output_mapping_weights
    
    def smoothL1(self, y_true, y_pred):
        x = tf.abs(y_true - y_pred)
        x = tf.where(x < self.HUBER_DELTA, 0.5*x**2, self.HUBER_DELTA*(x-0.5*self.HUBER_DELTA))
        return tf.reduce_sum(x)

    def trans_conv2d(self, batch_input, out_channels, kernel_size=3, strides=1):
        initializer = tf.truncated_normal_initializer(0, 0.01)
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=kernel_size, strides=strides,
            padding="same", kernel_initializer=initializer, kernel_regularizer = self.regularizer)

    def conv2d(self, batch_input, out_channels, kernel_size=3, strides=1, use_bias=False):
        initializer = tf.truncated_normal_initializer(0, 0.01)
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=kernel_size, strides=strides,
            padding="same", kernel_initializer=initializer, kernel_regularizer = self.regularizer, use_bias=use_bias)
        
    def conv2d_ws(self, batch_input, out_channels, kernel_size=3, strides=1):
#        w = tf.Variable(tf.truncated_normal([kernel_size*kernel_size*int(batch_input.get_shape()[-1]), out_channels], dtype=tf.float32))
#        w_mean = tf.reduce_mean(w, axis=0, keep_dims=True)
#        w -= w_mean
#        w_std = tf.keras.backend.std(w, axis=0, keepdims=True)
#        w = w / (w_std + 1e-5)
#        w = tf.reshape(w, [kernel_size, kernel_size, int(batch_input.get_shape()[-1]), out_channels])
        
        w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, int(batch_input.get_shape()[-1]), out_channels], dtype=tf.float32))
        w_mean = tf.reduce_mean(w, axis=[0, 1, 2], keep_dims=True)
        w -= w_mean
        w_std = tf.keras.backend.std(w, axis=[0, 1, 2], keepdims=True)
        w = w / (w_std + 1e-5)
        
        return tf.nn.conv2d(batch_input, w, [1, strides, strides, 1], "SAME")
        
        
    def batchnorm(self, batch_input):
        initializer = tf.truncated_normal_initializer(1.0, 0.01)
        return tf.layers.batch_normalization(batch_input, axis=-1, epsilon=1e-5, 
            momentum=0.95, training=self.is_training_pholder, gamma_initializer=initializer,
            gamma_regularizer = self.regularizer) 
    
    def conv_bn_relu(self, batch_input, out_channels, kernel_size=3, strides=1):
        conv = self.conv2d(batch_input, out_channels, kernel_size, strides)
        bn = self.batchnorm(conv)
        return tf.nn.relu(bn)
        
    def conv_bn(self, batch_input, out_channels, kernel_size=3, strides=1):
        conv = self.conv2d(batch_input, out_channels, kernel_size, strides)
        bn = self.batchnorm(conv)
        return bn        
    
    def residual(self, ins, num_out=None):
        ''' the bottleneck residual module
        Args:
            ins: the inputs
            k: kernel size
            num_out: number of the output feature madepth, default set as the same as input
        Returns:
            residual network output
        '''
        num_in = ins.shape[-1].value
        if num_out is None:
            num_out = num_in
        
        half_num_out = int(num_out//2)
        out_1 = self.conv_bn_relu(ins, half_num_out, kernel_size=1)
        out_1 = self.conv_bn_relu(out_1, half_num_out, kernel_size=3)
        out_1 = self.conv_bn(out_1, num_out, kernel_size=1)
    
        if num_out == num_in:
            out_2 = ins
        else:
            out_2 = self.conv_bn(ins, num_out, kernel_size=1)
        return tf.nn.relu(out_1+out_2)
        
    def udepthampling_nearest(self, inputs, scale):
        assert scale>1, 'scale of udepthampling should be larger then 1'
        new_h = int(inputs.shape[1]*scale)
        new_w = int(inputs.shape[2]*scale)
        return tf.image.resize_images(inputs, [new_h, new_w], 
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
    def hourglass(self, ins, n, kernel_size=3):
        ''' hourglass is created recursively, each time the module spatial resolution remains the same
        '''
        upper1 = self.residual(ins)
        
        k = kernel_size
        lower1 = tf.layers.max_pooling2d(ins, [k,k], strides=2, padding='same')
        lower1 = self.residual(lower1)
    
        if n > 1:
            lower2 = self.hourglass(lower1, n-1)
        else:
            lower2 = lower1
    
        lower3 = self.residual(lower2)
        upper2 = self.udepthampling_nearest(lower3, 2)
        print('[hourglass] n={}, shape={}'.format(n, upper1.shape))
    
        return upper1+upper2
        