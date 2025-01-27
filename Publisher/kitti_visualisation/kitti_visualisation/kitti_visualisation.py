from __future__ import print_function
import cv2
import numpy as np
from rclpy.node import Node
import os
import rclpy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import transforms3d
import time
from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2,PointField,Imu
import threading
from sensor_msgs_py import point_cloud2 as pcl2
from builtin_interfaces.msg import Duration as DurationMsg
import ament_index_python
import pandas as pd
from collections import deque

#项目所在路径
PROJECT_PATH='/home/admin/Project/ROS/'

#所需数据路径
DATA_PATH=os.path.join(PROJECT_PATH,'kitti-data/2011_09_26/2011_09_26_drive_0005_sync')
TRACKING_DATA_PATH=os.path.join(PROJECT_PATH,'kitti-data/training/label_02/0000.txt')
CALIB_PATH=os.path.join(PROJECT_PATH,'kitti-data/2011_09_26/')
MODEL_PATH = f'file:///home/admin/Project/ROS/kitti-data/CarModel/A.stl'

dtype=PointField.FLOAT32
fields = [PointField(name='x', offset=0, datatype=dtype, count=1),
          PointField(name='y', offset=4, datatype=dtype, count=1),
          PointField(name='z', offset=8, datatype=dtype, count=1),
          PointField(name='intensity', offset=12, datatype=dtype, count=1)]
IMU_COLUMN_NAMES=['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu','ax', 'ay', 'az', 'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode']
TRACKING_COLUMN_NAMES=['frame','track_id','type','truncated','occluded','alpha','bbox_left','bbox_top','bbox_right','bbox_bottom','height','width','length','pos_x','pos_y','pos_z','rot_y']
DETECTION_COLOR_DICT={'Car':(255,255,0),'Pedestrian':(0,255,255),'Cyclist':(255,0,255),'Van':(255,100,100),'DontCare':(100,100,100)}
LINES=[[0,1],[1,2],[2,3],[3,0]]
LINES+=[[4,5],[5,6],[6,7],[7,4]]
LINES+=[[4,0],[5,1],[6,2],[7,3]]
LINES+=[[4,1],[5,0]]
LIFETIME=0.05
FRAME_ID='map'
ct=time.time()


#######################################
#以下涉及坐标系转换的部分(48-424)来自
#https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py
class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];
    #print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2,:]<0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2,:]<0.1):
      orientation_2d = None
      return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
    return image
#到此结束
#######################################


class NodePublisher(Node):#创建节点的对象
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("Node created!")

class Object():#储存绘制轨迹的点序列的对象
    def __init__(self,center):
        self.locations=deque(maxlen=200)
        self.locations.appendleft(center)
    def update(self,center,displacement,yaw_change):
        for i in range(len(self.locations)):
            x0,y0=self.locations[i]
            x1=x0*np.cos(yaw_change)+y0*np.sin(yaw_change)-displacement
            y1=-x0*np.sin(yaw_change)+y0*np.cos(yaw_change)
            self.locations[i]=np.array([x1,y1])
        if center is not None:
            self.locations.appendleft(center)
    def reset(self):
        self.locations=deque(maxlen=200)

def read_camera(path):#读取图片数据
    return cv2.imread(path)

def read_point_cloud(path):#读取点云数据
    return np.fromfile(path,dtype=np.float32).reshape(-1,4)

def read_imu(path):#读取IMU数据
    df=pd.read_csv(path,header=None,sep=' ')
    df.columns=IMU_COLUMN_NAMES
    return df

def read_tracking(path):#读取打标数据
    df=pd.read_csv(path,header=None,sep=' ')
    df.columns=TRACKING_COLUMN_NAMES
    df.loc[df.type.isin(['Truck','Tram']),'type']='Car'
    df=df[df.type.isin(['Car','Pedestrian','Cyclist','Van','DontCare'])]
    return df

def publish_camera(cam_pub,bridge,image,boxes,types):#发布图片并绘制2d侦测框
    for typ,box in zip(types,boxes):
        top_left=int(box[0]),int(box[1])
        bottom_right=int(box[2]),int(box[3])
        cv2.rectangle(image,top_left,bottom_right,DETECTION_COLOR_DICT[typ],1)
    cam_pub.publish(bridge.cv2_to_imgmsg(image,'bgr8'))

def publish_point_cloud(pcl_pub,point_cloud):#发布点云
    header=Header()
    header.stamp.sec=int(ct)
    header.stamp.nanosec=int((ct%1)*1e9)
    header.frame_id=FRAME_ID
    pcl_pub.publish(pcl2.create_cloud(header,fields,point_cloud[:,:4]))

def publish_car_model(model_pub):#发布车模型
    mesh_marker=Marker()
    mesh_marker.header.frame_id=FRAME_ID
    mesh_marker.header.stamp.sec=int(ct)
    mesh_marker.header.stamp.nanosec=int((ct%1)*1e9)
    mesh_marker.id=-1
    duration_msg=DurationMsg()
    duration_msg.sec=0
    duration_msg.nanosec=0
    mesh_marker.lifetime=duration_msg
    mesh_marker.type=Marker.MESH_RESOURCE
    mesh_marker.mesh_resource=MODEL_PATH
    mesh_marker.pose.position.x=0.0
    mesh_marker.pose.position.y=0.0
    mesh_marker.pose.position.z=-1.3
    q=transforms3d.euler.euler2quat(0,0,np.pi)
    mesh_marker.pose.orientation.x=q[1]
    mesh_marker.pose.orientation.y=q[2]
    mesh_marker.pose.orientation.z=q[3]
    mesh_marker.pose.orientation.w=q[0]
    mesh_marker.color.r=0.25
    mesh_marker.color.g=1.0
    mesh_marker.color.b=1.0
    mesh_marker.color.a=1.0
    mesh_marker.scale.x=1.25
    mesh_marker.scale.y=1.25
    mesh_marker.scale.z=1.25
    model_pub.publish(mesh_marker)

def publish_ego_car(ego_car_pub):#发布摄像机视角
    marker=Marker()
    marker.header.frame_id=FRAME_ID
    marker.header.stamp.sec=int(ct)
    marker.header.stamp.nanosec=int((ct%1)*1e9)
    marker.id=0
    marker.action=Marker.ADD
    duration_msg=DurationMsg()
    duration_msg.sec=0
    duration_msg.nanosec=0
    marker.lifetime=duration_msg
    marker.type=Marker.LINE_STRIP
    marker.color.r=0.0
    marker.color.g=1.0
    marker.color.b=1.0
    marker.color.a=1.0
    marker.scale.x=0.1
    marker.points=[]
    marker.points.append(Point(x=20.0,y=-20.0,z=0.0))
    marker.points.append(Point(x=0.0,y=0.0,z=0.0))
    marker.points.append(Point(x=20.0,y=20.0,z=0.0))
    ego_car_pub.publish(marker)

def publish_imu(imu_pub,imu_data):#发布IMU数据
    imu=Imu()
    imu.header.frame_id=FRAME_ID
    imu.header.stamp.sec=int(ct)
    imu.header.stamp.nanosec=int((ct%1)*1e9)
    q=transforms3d.euler.euler2quat(float(imu_data.roll.iloc[0]),float(imu_data.pitch.iloc[0]),float(imu_data.yaw.iloc[0]))
    imu.orientation.w=q[0]
    imu.orientation.x=q[1]
    imu.orientation.y=q[2]
    imu.orientation.z=q[3]
    imu.linear_acceleration.x=imu_data.af.iloc[0]
    imu.linear_acceleration.y=imu_data.al.iloc[0]
    imu.linear_acceleration.z=imu_data.au.iloc[0]
    imu.angular_velocity.x=imu_data.wf.iloc[0]
    imu.angular_velocity.y=imu_data.wl.iloc[0]
    imu.angular_velocity.z=imu_data.wu.iloc[0]
    imu_pub.publish(imu)

def publish_3dbox(box3d_pub,corners_3d_velos,types,track_ids):#发布3d侦测框
    marker_array=MarkerArray()
    for i,corners_3d_velo in enumerate(corners_3d_velos):
        marker=Marker()
        marker.header.frame_id=FRAME_ID
        marker.header.stamp.sec=int(ct)
        marker.header.stamp.nanosec=int((ct%1)*1e9)
        marker.id=i
        marker.action=Marker.ADD
        duration_msg=DurationMsg()
        duration_msg.sec=0
        duration_msg.nanosec=int(LIFETIME*1e9)
        marker.lifetime=duration_msg
        marker.type=Marker.LINE_LIST
        b,g,r=DETECTION_COLOR_DICT[types[i]]
        marker.color.r=r/255.0
        marker.color.g=g/255.0
        marker.color.b=b/255.0
        marker.color.a=1.0
        marker.scale.x=0.05
        marker.points=[]
        for l in LINES:
            p1=corners_3d_velo[l[0]]
            marker.points.append(Point(x=p1[0],y=p1[1],z=p1[2]))
            p2=corners_3d_velo[l[1]]
            marker.points.append(Point(x=p2[0],y=p2[1],z=p2[2]))
        marker_array.markers.append(marker)
        text_marker=Marker()
        text_marker.header.frame_id=FRAME_ID
        text_marker.header.stamp.sec=int(ct)
        text_marker.header.stamp.nanosec=int((ct%1)*1e9)
        text_marker.id=i+1000
        duration_msg=DurationMsg()
        duration_msg.sec=0
        duration_msg.nanosec=int(LIFETIME*1e9)
        text_marker.lifetime=duration_msg
        text_marker.action=Marker.ADD
        text_marker.type=Marker.TEXT_VIEW_FACING
        p4=np.mean(corners_3d_velo,axis=0)
        text_marker.pose.position.x=float(p4[0])
        text_marker.pose.position.y=float(p4[1])
        text_marker.pose.position.z=float(p4[2]+1.0)
        text_marker.text=str(track_ids[i])+' '+types[i]
        text_marker.scale.x=0.7
        text_marker.scale.y=0.7
        text_marker.scale.z=0.7
        b,g,r=DETECTION_COLOR_DICT[types[i]]
        text_marker.color.r=r/255.0
        text_marker.color.g=g/255.0
        text_marker.color.b=b/255.0
        text_marker.color.a=1.0
        marker_array.markers.append(text_marker)
    box3d_pub.publish(marker_array)

def publish_loc(loc_pub,tracker,centers):#发布物体轨迹
    marker_array=MarkerArray()
    for track_id in centers:
        marker=Marker()
        marker.header.frame_id=FRAME_ID
        marker.header.stamp.sec=int(ct)
        marker.header.stamp.nanosec=int((ct%1)*1e9)
        marker.action=Marker.ADD
        duration_msg=DurationMsg()
        duration_msg.sec=0
        duration_msg.nanosec=int(LIFETIME*1e9)
        marker.lifetime=duration_msg
        marker.type=Marker.LINE_STRIP
        marker.id=int(track_id+10000)
        marker.color.r=0.0
        marker.color.g=1.0
        marker.color.b=1.0
        marker.color.a=1.0
        marker.scale.x=0.1
        marker.points=[]
        for p in tracker[track_id].locations:
            marker.points.append(Point(x=float(p[0]),y=float(p[1]),z=0.0))
        marker_array.markers.append(marker)
    loc_pub.publish(marker_array)


def compute_3d_box_cam2(h,w,l,x,y,z,yaw):#计算3d侦测框坐标
    R=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    x_corners=[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners=[0,0,0,0,-h,-h,-h,-h]
    z_corners=[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2=np.dot(R,np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2+=np.vstack([x,y,z])
    return corners_3d_cam2

def main():#主函数
    event=threading.Event()
    frame=0
    rclpy.init()
    node=NodePublisher('CarModelPublisher')
    cam_pub=node.create_publisher(Image,'ImagePub',10)
    pcl_pub=node.create_publisher(PointCloud2,'PointCloudPub',10)
    ego_pub=node.create_publisher(Marker,'EgoCarPub',10)
    model_pub=node.create_publisher(Marker,'CarModelPub',10)
    imu_pub=node.create_publisher(Imu,'ImuPub',10)
    box3d_pub=node.create_publisher(MarkerArray,'Box3dPub',10)
    loc_pub=node.create_publisher(MarkerArray,'LocPub',10)
    bridge=CvBridge()
    df_tracking=read_tracking(TRACKING_DATA_PATH)
    calib=Calibration(CALIB_PATH,from_video=True)
    tracker={}
    prev_imu_data=None
    while True:
        boxes=np.array(df_tracking[df_tracking.frame==frame][['bbox_left','bbox_top','bbox_right','bbox_bottom']])
        boxes_3d=np.array(df_tracking[df_tracking.frame==frame][['height','width','length','pos_x','pos_y','pos_z','rot_y']])
        types=np.array(df_tracking[df_tracking.frame==frame]['type'])
        track_ids=np.array(df_tracking[df_tracking.frame==frame]['track_id'])
        corners_3d_velos=[]
        centers={}
        for track_id,box_3d in zip(track_ids,boxes_3d):
            corners_3d_cam2=compute_3d_box_cam2(*box_3d)
            corners_3d_velo=calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos+=[corners_3d_velo]
            centers[track_id]=np.mean(corners_3d_velo,axis=0)[:2]
        centers[-1]=np.array([0,0])
        image=read_camera(os.path.join(DATA_PATH,'image_02/data/%010d.png'%frame))
        publish_camera(cam_pub,bridge,image,boxes,types)
        point_cloud=read_point_cloud(os.path.join(DATA_PATH,'velodyne_points/data/%010d.bin'%frame))
        publish_point_cloud(pcl_pub,point_cloud)
        imu_data=read_imu(os.path.join(DATA_PATH,'oxts/data/%010d.txt'%frame))
        publish_imu(imu_pub,imu_data)
        publish_car_model(model_pub)
        publish_ego_car(ego_pub)
        publish_3dbox(box3d_pub,corners_3d_velos,types,track_ids)
        if prev_imu_data is None:
            for track_id in centers:
                tracker[track_id]=Object(centers[track_id])
        else:
            displacement=0.1*np.linalg.norm(imu_data[['vf','vl']])
            yaw_change=float((imu_data.yaw.iloc[0])-(prev_imu_data.yaw.iloc[0]))
            for track_id in centers:
                if track_id in tracker:
                    tracker[track_id].update(centers[track_id],displacement,yaw_change)
                else:
                    tracker[track_id]=Object(centers[track_id])
            for track_id in tracker:
                if track_id not in centers:
                    tracker[track_id].update(None,displacement,yaw_change)
        prev_imu_data=imu_data
        publish_loc(loc_pub,tracker,centers)
        event.wait(0.033)
        frame+=1
        if frame == 154:
            frame=0
            for track_id in tracker:
                tracker[track_id].reset()
    print('Resrt!')

if __name__=='__main__':
    main()
