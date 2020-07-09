import os
import sys
sys.path.append(os.getcwd()+'/data/')
sys.path.append(os.getcwd()+'/network/')
sys.path.append(os.getcwd()+'/util/')

from data.stream import MultiDataStream
from data.importers import MSRA15Importer
from util.preprocess import norm_dm
import tensorflow as tf
import argparse
import numpy as np
import time

rng = np.random.RandomState(23455)

from network.dense_offset_graph import Model

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# images in batch')
parser.add_argument('--image_size', dest='image_size', type=int, default=96, help='size of image')
parser.add_argument('--image_c_dim', dest='image_c_dim', type=int, default=1, help='# of input image channels')
parser.add_argument('--pose_dim', dest='pose_dim', type=int, default=63, help='dimentions of hand pose')
parser.add_argument('--num_jnt', dest='num_jnt', type=int, default=21, help='# of hand joints')
parser.add_argument('--log_dir', dest='log_dir', default='./log/MSRA', help='logs are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint/MSRA/dense_offset_graph_A', help='models are saved here')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=58, help='# of epoch')
parser.add_argument('--dataset', dest='dataset', default='MSRA', help='dataset')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--num_fea', dest='num_fea', type=int, default=128, help='# of feature maps of hourglass')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--test_sub', type=int, default = 3)
parser.add_argument('--data_root', dest='data_root', default='/home/data/3D_hand_pose_estimation/MSRA/cvpr15_MSRAHandGestureDB/', help='dataset are saved here')

args = parser.parse_args()

def main(_):
    start = time.clock()

    if args.dataset == 'MSRA':
        di = MSRA15Importer(args.data_root, cacheDir='./cache/MSRA/',refineNet=None)
        Seq_all = []
        MID = args.test_sub
        for seq in range(9):     
            if seq == MID:
                Seq_train_ = di.loadSequence('P{}'.format(seq), rng=rng, shuffle=False, docom=False, cube=(175, 175, 175))
            else:
                Seq_train_ = di.loadSequence('P{}'.format(seq), rng=rng, shuffle=False, docom=False, cube=None)
            Seq_all.append(Seq_train_)

        Seq_test_raw = Seq_all.pop(MID)
        Seq_test = Seq_test_raw.data
        Seq_train = [seq_data for seq_ in Seq_all for seq_data in seq_.data]

        train_num = len(Seq_train)
        print ('loaded over with %d train samples'%train_num)       
        imgs = np.asarray([d.dpt.copy() for d in Seq_train], 'float32')
        gt3Dcrops = np.asarray([d.gt3Dcrop for d in Seq_train], dtype='float32')
        M = np.asarray([d.T for d in Seq_train], dtype='float32')
        com2D = np.asarray([d.com2D for d in Seq_train], 'float32')
        cube = np.asarray([d.cube for d in Seq_train], 'float32')
        # uv_crop = np.asarray([d.gtcrop for d in Seq_train], dtype='float32')[:, :, 0:-1]
        del Seq_train

        train_stream = MultiDataStream([imgs, gt3Dcrops, M, com2D, cube])
    else:
        raise ValueError('error dataset %s'%args.dataset)
    
    test_num=len(Seq_test)
    print ('loaded over with %d test samples'%test_num) 
    test_gt3Dcrops = np.asarray([d.gt3Dcrop for d in Seq_test], dtype='float32')
    test_M = np.asarray([d.T for d in Seq_test], dtype='float32')
    # test_com2D = np.asarray([d.com2D for d in Seq_test], 'float32')  
    # test_uv_crop = np.asarray([d.gtcrop for d in Seq_test], dtype='float32')[:, :, 0:-1]
    test_uv = np.asarray([d.gtorig for d in Seq_test], 'float32')[:, :, 0:-1]
    test_com3D = np.asarray([d.com3D for d in Seq_test], 'float32') 
    test_cube = np.asarray([d.cube for d in Seq_test], 'float32')
    test_imgs = np.asarray([d.dpt.copy() for d in Seq_test], 'float32')
    test_data=np.ones_like(test_imgs)
    for it in range(test_num):
        test_data[it]=norm_dm(test_imgs[it], test_com3D[it], test_cube[it])
    del Seq_test    
    test_stream = MultiDataStream([test_data, test_gt3Dcrops, test_M, test_com3D, test_uv, test_cube])
    clip_index = np.int(np.floor(test_num/args.batch_size)) * args.batch_size
    extra_test_data = [test_data[clip_index:], test_gt3Dcrops[clip_index:], test_M[clip_index:], 
                       test_com3D[clip_index:], test_uv[clip_index:], test_cube[clip_index:]]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    tf.set_random_seed(1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = Model(sess, args)
        model.train(args, train_stream, test_stream) if args.phase == 'train' \
            else model.test(args, test_stream, extra_test_data=None)
        end = time.clock()
        print ('running time: %f s'%(end-start))

if __name__ == '__main__':
    tf.app.run()