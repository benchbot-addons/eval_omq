
# Software License Agreement (BSD License)

# Modified by Suman Raj Bista 
# for Robotic Vision Evaluation and Benchmarking Project at ACRV/QUT
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of ACRV/QUT nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
from shapely.geometry import Polygon


class IoU:

    def __init__(self):
        """ initilise with args
        """
        
        self.__gt_bb = []
        self.__est_bb = []
        self.__gtd = []
        self.__gtc = []
        self.__ed = []
        self.__ec = []



    def __rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s,  0],
                        [s,  c,  0],
                        [0,  0,  1]])



    def get_bbox3D(self, box_size, heading_angle, center):
        """ Calculate 3D bounding box corners from its parameterization.
        Input:
            box_size: (length,wide,height) , heading_angle: rad , center:  (x,y,z)
        Output:
            corners3D:  3D box corners, vol: volume of 3D bounding box
        """

        l,w,h = box_size
        #print(box_size)
        vol = (l)*(w)*(h)
        
        R = self.__rotz(heading_angle)

        corners3D = 0.5 * np.array([[-l , -l,  l , l , -l , -l , l , l ],
                                    [w , -w , -w , w , w , -w , -w , w ],
                                    [-h , -h , -h , -h , h , h , h , h ],
                                   ])
        
        #corners3D = np.dot(R,corners3D)

        corners3D[0,:] = corners3D[0,:] + center[0]
        corners3D[1,:] = corners3D[1,:] + center[1]
        corners3D[2,:] = corners3D[2,:] + center[2]
        
        #print(corners3D)
        return corners3D, vol



    def get_boundingCuboid(self, extent, centroid):
        """ Calculate 3D bounding box corners from centre and 3 dimensions
        """
        corners3D, vol = self.get_bbox3D(extent,0.0,centroid)
        return  corners3D



    def get_boundingbox_bev(self, bbox_3D):
        return Polygon(np.transpose(bbox_3D[(0,1), 0:4]))

    def __get_boundingVolume(self, bbox_3D):
        l =  np.max(bbox_3D[0]) - np.min(bbox_3D[0])
        b =  np.max(bbox_3D[1]) - np.min(bbox_3D[1])
        h =  np.max(bbox_3D[2]) - np.min(bbox_3D[2])
        return l*b*h

    
    def __get_ovelapHeight(self, bbox_3Da, bbox_3Db):
        """
        Calculate overlapping height between two 3D bounding box
        Input: 8 corners of bounding box 
        Output: overlapping height
        """

        h_a_min = np.min(bbox_3Da[2])
        h_a_max = np.max(bbox_3Da[2])

        h_b_min = np.min(bbox_3Db[2])
        h_b_max = np.max(bbox_3Db[2])

        max_of_min = np.max([h_a_min, h_b_min])
        min_of_max = np.min([h_a_max, h_b_max])

        hoverlap = np.max([0, min_of_max - max_of_min])
        #print("h_overlap = ", hoverlap)

        return hoverlap

    

    def __get_intersestionin2D(self, poly_a, poly_b):
        """
        Calculate overlapping area between two polygons
        Input: 2 polygons in 2d
        Output: overlapping polygon and area of overlap
        """
        poly_xy_int = poly_a.intersection(poly_b)
        return poly_xy_int, poly_xy_int.area
    


    def __get_polygonXY(self, bbox3D):
        """
        birdeye view of 3d bbox
        Input: 3d box
        Output: projection of 3d bbox in xyplane and area of the polygon in xy
        """
      
        poly_xy = Polygon(zip(*bbox3D[0:2, 0:4]))
        return poly_xy, poly_xy.area



    def __iou(self, set_a, set_b, set_intersect):
        union = set_a + set_b - set_intersect
        return set_intersect / union if union else 0.



    def cal_IoU(self, bbox_3Da, bbox_3Db):

        vol_a = self.__get_boundingVolume(bbox_3Da)
        vol_b = self.__get_boundingVolume(bbox_3Db)

        poly_a, area_a = self.__get_polygonXY(bbox_3Da)
        poly_b, area_b = self.__get_polygonXY(bbox_3Db)

        poly_xy, area_xy = self.__get_intersestionin2D(poly_a, poly_b)

        h_overlap = self.__get_ovelapHeight(bbox_3Da, bbox_3Db)

        vol_ab =  h_overlap * area_xy

        iou_bev = self.__iou(area_a, area_b, area_xy)
        iou_3D = self.__iou(vol_a, vol_b, vol_ab)
       
        return iou_bev, iou_3D

    
    def calculate(self, box_size_gt, center_gt, box_size_est, center_est):

        corners_gt, v_gt =   self.get_bbox3D(box_size_gt, 0.0 ,center_gt)
        corners_est, v_est =   self.get_bbox3D(box_size_est, 0.0 ,center_est)
        

        poly_gt, area_gt = self.__get_polygonXY(corners_gt)
        poly_est, area_est = self.__get_polygonXY(corners_est)

        poly_int_xy, area_int_xy = self.__get_intersestionin2D(poly_gt, poly_est)

        h_int = self.__get_ovelapHeight(corners_gt, corners_est)

        vol_int = h_int *  area_int_xy

        iou_bev = self.__iou(area_gt, area_est, area_int_xy)
        iou_3D = self.__iou(v_gt, v_est, vol_int)

        
        #print(v_gt, v_est, vol_int, area_gt, area_est, area_int_xy, h_int)
       
        return iou_bev, iou_3D

    
    def dict_iou(self, dict1, dict2):
        corners1, v1 =   self.get_bbox3D(dict1['extent'], 0.0 ,dict1['centroid'])
        corners2, v2 =   self.get_bbox3D(dict2['extent'], 0.0 ,dict2['centroid'])
        

        poly1, area1 = self.__get_polygonXY(corners1)
        poly2, area2 = self.__get_polygonXY(corners2)

        poly_int_xy, area_int_xy = self.__get_intersestionin2D(poly1, poly2)

        h_int = self.__get_ovelapHeight(corners1, corners2)

        vol_int = h_int *  area_int_xy

        iou_bev = self.__iou(area1, area2, area_int_xy)
        iou_3D = self.__iou(v1, v2, vol_int)

        
        #print(v_gt, v_est, vol_int, area_gt, area_est, area_int_xy, h_int)
       
        return iou_bev, iou_3D

    def dict_prop_fraction(self, prop_dict, gt_dict):
        """
        Function for determining the proportion of a proposal is covered by the ground-truth
        :param prop_dict:
        :param gt_dict:
        :return:
        """
        corners1, v1 = self.get_bbox3D(prop_dict['extent'], 0.0, prop_dict['centroid'])
        corners2, v2 = self.get_bbox3D(gt_dict['extent'], 0.0, gt_dict['centroid'])

        poly1, area1 = self.__get_polygonXY(corners1)
        poly2, area2 = self.__get_polygonXY(corners2)

        poly_int_xy, area_int_xy = self.__get_intersestionin2D(poly1, poly2)

        h_int = self.__get_ovelapHeight(corners1, corners2)

        vol_int = h_int * area_int_xy

        return vol_int / v1




    

   





