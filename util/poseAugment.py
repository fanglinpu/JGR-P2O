import numpy
import numpy as np
import math
import cv2
from collections import namedtuple

CameraOption = namedtuple('CameraOption', ['focal_x', 'focal_y', 'center_x', 'center_y', 'width', 'height', 'far_point'])

class Camera(object):
    def __init__(self, dataset):
        intel = [241.42, 241.42, 160, 120, 320, 240, 32001]
        kinect = [588.235, 587.084, 320, 240, 640, 480, 2001]
        
        #set as default
        if dataset == 'NYU':
            current = CameraOption(*kinect)
        elif dataset == 'ICVL':
            current = CameraOption(*intel)
        elif dataset == 'MSRA':
            current = CameraOption(*intel)
        else:
            print (dataset)
            raise NotImplementedError('Unknown dataset %s'%dataset)
    
        self.focal_x = current.focal_x
        self.focal_y = current.focal_y
        self.center_x = current.center_x
        self.center_y = current.center_y
        self.width = current.width
        self.height = current.height
        self.far_point = current.far_point

    def to3D(self, pt2):
        pt3 = np.zeros((3), np.float32)
        pt3[0] = (pt2[0] - self.center_x)*pt2[2] / self.focal_x
        pt3[1] = (self.center_y - pt2[1])*pt2[2] / self.focal_y
        #pt3[1] = (pt2[1] - self.center_y)*pt2[2] / self.focal_y
        pt3[2] = pt2[2]
        return pt3

    def to2D(self, pt3):
        pt2 = np.zeros((3), np.float32)
        pt2[0] =  pt3[0]*self.focal_x / pt3[2] + self.center_x
        pt2[1] = -pt3[1]*self.focal_y / pt3[2] + self.center_y
        #pt2[1] = pt3[1]*self.focal_y / pt3[2] + self.center_y
        pt2[2] = pt3[2]
        return pt2
           
    def to3D_v(self, pt2s):
        pt3s = np.zeros((pt2s.shape[0], 3), np.float32)
        for i in range(pt2s.shape[0]):
            pt3s[i] = self.to3D(pt2s[i])
        return pt3s  
        
    def to2D_v(self, pt3s):
        pt2s = np.zeros((pt3s.shape[0], 3), np.float32)
        for i in range(pt3s.shape[0]):
            pt2s[i] = self.to2D(pt3s[i])
        return pt2s

class PoseAugment(object):
    def __init__(self, dataset):
        self.Camera = Camera(dataset)
	
    def transformPoint2D(self, pt, M):
        """
        Transform point in 2D coordinates
        :param pt: point coordinates
        :param M: transformation matrix
        :return: transformed point
        """
        pt2 = numpy.dot(numpy.asarray(M).reshape((3, 3)), numpy.asarray([pt[0], pt[1], 1]))
        return numpy.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])

    def transformPoints2D(self, pts, M):
        """
        Transform points in 2D coordinates
        :param pts: point coordinates
        :param M: transformation matrix
        :return: transformed points
        """
        ret = pts.copy()
        for i in range(pts.shape[0]):
            ret[i, 0:2] = self.transformPoint2D(pts[i, 0:2], M)
        return ret
    
    def comToBounds(self, com, size):
            """
            Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
            :param com: center of mass, in image coordinates (x,y,z), z in mm
            :param size: (x,y,z) extent of the source crop volume in mm
            :return: xstart, xend, ystart, yend, zstart, zend
            """
            if numpy.isclose(com[2], 0.):
                print ("Warning: CoM ill-defined!")
                xstart = 640//4
                xend = xstart + 640//2
                ystart = 480//4
                yend = ystart + 480//2
                zstart = 600
                zend = 900
            else:
                zstart = com[2] - size[2] / 2.
                zend = com[2] + size[2] / 2.
                xstart = int(math.floor((com[0] * com[2] / self.Camera.focal_x - size[0] / 2.) / com[2]*self.Camera.focal_x))
                xend = int(math.floor((com[0] * com[2] / self.Camera.focal_x + size[0] / 2.) / com[2]*self.Camera.focal_x))
                ystart = int(math.floor((com[1] * com[2] / self.Camera.focal_y - size[1] / 2.) / com[2]*self.Camera.focal_y))
                yend = int(math.floor((com[1] * com[2] / self.Camera.focal_y + size[1] / 2.) / com[2]*self.Camera.focal_y))
            return xstart, xend, ystart, yend, zstart, zend
    
    def comToTransform(self, com, size, dsize):
            """
            Calculate affine transform from crop
            :param com: center of mass, in image coordinates (x,y,z), z in mm
            :param size: (x,y,z) extent of the source crop volume in mm
            :return: affine transform
            """
    
            xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size)
    
            trans = numpy.eye(3)
            trans[0, 2] = -xstart
            trans[1, 2] = -ystart
    
            wb = (xend - xstart)
            hb = (yend - ystart)
            if wb > hb:
                scale = numpy.eye(3) * dsize[0] / float(wb)
                sz = (dsize[0], hb * dsize[0] / wb)
            else:
                scale = numpy.eye(3) * dsize[1] / float(hb)
                sz = (wb * dsize[1] / hb, dsize[1])
            scale[2, 2] = 1
    
            xstart = int(numpy.floor(dsize[0] / 2. - sz[1] / 2.))
            ystart = int(numpy.floor(dsize[1] / 2. - sz[0] / 2.))
            off = numpy.eye(3)
            off[0, 2] = xstart
            off[1, 2] = ystart
    
            return numpy.dot(off, numpy.dot(scale, trans))        
    
    def rotatePoint2D(self, p1, center, angle):
        """
        Rotate a point in 2D around center
        :param p1: point in 2D (u,v,d)
        :param center: 2D center of rotation
        :param angle: angle in deg
        :return: rotated point
        """
        alpha = angle * numpy.pi / 180.
        pp = p1.copy()
        pp[0:2] -= center[0:2]
        pr = numpy.zeros_like(pp)
        pr[0] = pp[0]*numpy.cos(alpha) - pp[1]*numpy.sin(alpha)
        pr[1] = pp[0]*numpy.sin(alpha) + pp[1]*numpy.cos(alpha)
        pr[2] = pp[2]
        ps = pr
        ps[0:2] += center[0:2]
        return ps        
            
    def moveCoM(self, dpt, com, off, joints3D, M, cube, pad_value=0.):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """
    
        # if offset is 0, nothing to do
        if numpy.allclose(off, 0.):
            com3D = self.Camera.to3D(com)
            joint_2D = self.Camera.to2D_v(joints3D + com3D)
            joints2D = self.transformPoints2D(joint_2D[:, 0:2], M)
            return dpt, joints3D, joints2D, com, M, com3D
    
        # add offset to com
        new_com = self.Camera.to2D(self.Camera.to3D(com) + off)
    
        # check for 1/0.
        if not (numpy.allclose(com[2], 0.) or numpy.allclose(new_com[2], 0.)):
            # scale to original size
            Mnew = self.comToTransform(new_com, cube, dpt.shape)            
            warped = cv2.warpPerspective(dpt, numpy.dot(Mnew, numpy.linalg.inv(M)), dpt.shape, flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)          
            warped[numpy.isclose(warped, 32000.)] = pad_value
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, cube)
            msk1 = numpy.logical_and(warped < zstart, warped != 0)
            msk2 = numpy.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later
            new_dpt = warped.copy()
        else:
            Mnew = M
            new_dpt = dpt
        
        com3D = self.Camera.to3D(com)
        new_com3D = self.Camera.to3D(new_com)
        joint_2D = self.Camera.to2D_v(joints3D + com3D)
            
        # adjust joint positions to new CoM
        new_joints3D = joints3D + com3D - new_com3D
        
        # adjust joint positions to new crop
        new_joints2D = self.transformPoints2D(joint_2D[:, 0:2], Mnew)
    
        return new_dpt, new_joints3D, new_joints2D, new_com, Mnew, new_com3D
    
    def rotateHand(self, dpt, com, rot, joints3D, M, cube, pad_value=0.):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param uv: 2D joint coordinates in original depth image
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """
    
        # if rot is 0, nothing to do
        if numpy.allclose(rot, 0.):
            com3D = self.Camera.to3D(com)
            joint_2D = self.Camera.to2D_v(joints3D + com3D)
            joints2D = self.transformPoints2D(joint_2D[:, 0:2], M)
            return dpt, joints3D, joints2D, com, M, com3D
    
        rot = numpy.mod(rot, 360)
    
        M_affine = cv2.getRotationMatrix2D((dpt.shape[1]//2, dpt.shape[0]//2), -rot, 1)
                
        new_dpt = cv2.warpAffine(dpt, M_affine, (dpt.shape[1], dpt.shape[0]), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)
    
        com3D = self.Camera.to3D(com)
        joint_2D = self.Camera.to2D_v(joints3D + com3D)
        data_2D = numpy.zeros_like(joint_2D)
        for k in range(data_2D.shape[0]):
            data_2D[k] = self.rotatePoint2D(joint_2D[k], com[0:2], rot)
        
        new_joints3D = self.Camera.to3D_v(data_2D) - com3D
        
        new_joints2D = self.transformPoints2D(data_2D[:, 0:2], M)
        
        new_com = com
        
        Mnew = M
    
        return new_dpt, new_joints3D, new_joints2D, new_com, Mnew, com3D
    
    
    def scaleHand(self, dpt, com, sc, joints3D, M, cube, pad_value=0.):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """
    
        # if scale is 1, nothing to do
        if numpy.allclose(sc, 1.):
            com3D = self.Camera.to3D(com)
            joint_2D = self.Camera.to2D_v(joints3D + com3D)
            joints2D = self.transformPoints2D(joint_2D[:, 0:2], M)
            return dpt, joints3D, joints2D, com, M, com3D
    
        new_cube = [s*sc for s in cube]
    
        # check for 1/0.
        if not numpy.allclose(com[2], 0.):
            # scale to original size
            Mnew = self.comToTransform(com, new_cube, dpt.shape)

            warped = cv2.warpPerspective(dpt, numpy.dot(Mnew, numpy.linalg.inv(M)), dpt.shape, flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)
            warped[numpy.isclose(warped, 32000.)] = pad_value
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, cube)
            msk1 = numpy.logical_and(warped < zstart, warped != 0)
            msk2 = numpy.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later
            new_dpt = warped.copy()
        else:
            Mnew = M
            new_dpt = dpt
    
        com3D = self.Camera.to3D(com)
        joint_2D = self.Camera.to2D_v(joints3D + com3D)
        
        new_joints3D = joints3D / sc
        
        new_joints2D = self.transformPoints2D(joint_2D[:, 0:2], Mnew)
        
        new_com = com
        
        return new_dpt, new_joints3D, new_joints2D, new_com, Mnew, com3D

