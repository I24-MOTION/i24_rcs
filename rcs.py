"""
This file supercedes all older versions of homography and coordinate system. It defines an 
object for performing image to state plane conversions via homography and state plane
to roadway coordinate system conversions via spline curvilinear coordinate system conversion.

This object implements only initialization and conversion functions - tests and utils are in other scripts
"""


import os
import _pickle as pickle
import pandas as pd
import numpy as np
import torch
import glob
import cv2
import time
import string
import re
import copy
import sys
import csv
import matplotlib.pyplot as plt
import json

from scipy import interpolate

try:
    
    import pyproj
    from pyproj import Proj, transform
except ModuleNotFoundError:
    print("Warning: no pyproj package detected, GPS conversion not enabled")
    
class I24_RCS:
    


#%% Utility functions
    def line_to_point(line,point):
        """
        Given a line defined by two points, finds the distance from that line to the third point
        line - (x0,y0,x1,y1) as floats
        point - (x,y) as floats
        Returns
        -------
        distance - float >= 0
        """
        
        numerator = np.abs((line[2]-line[0])*(line[1]-point[1]) - (line[3]-line[1])*(line[0]-point[0]))
        denominator = np.sqrt((line[2]-line[0])**2 +(line[3]-line[1])**2)
        
        return numerator / (denominator + 1e-08)
        
    def safe_name(func):
        """
        Wrapper function, catches camera names that aren't capitalized 
        """
        
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError:
                #print(args,kwargs)
                if type(kwargs["name"]) == list:
                    kwargs["name"] = [item.upper() for item in kwargs["name"]]
                elif type(kwargs["name"]) == str:
                    kwargs["name"] = kwargs["name"].upper()
                return func(*args, **kwargs)
        return new_func
    
    
    #%% Initialization and Setup Functions
    """
    
    3 coordinate systems are utilized in Curvilinear_Homography:
        - image coordinates
        - space coordinates (state plane coordinates) in feet
        - roadway coordianates / curvilinear coordinates in feet
    
    After fitting, each value of self.correspondence contains:
        H     - np array of size [3,3] used for image to space perspective transform
        H_inv - used for space to image perspective transform on ground plane
        P     - np array of size [3,4] used for space to image transform
        corr_pts - list of [x,y] points in image space that are fit for transform
        space_pts - corresponding list of [x,y] points in space (state plane coordinates in feet)
        state_plane_pts - same as space_pts but [x,y,id] (name is included)
        vps - z vanishing point [x,y] in image coordinates
        extents - xmin,xmax,ymin,ymax in roadway coordinates
        extents_space - list of array of [x,y] points defining boundary in state plane coordinates
        
    """
    
    
    def __init__(self, 
                 save_path,
                 aerial_ref_dir = None,
                 im_ref_dir = None,
                 downsample = 1,
                 default = "dynamic"):
        """
        Initializes homography object.
        
        aerial_ref_dir -None or str - path to directory with csv files of attributes labeled in space coordinates
        im_ref_dir   - None or str - path to directory with cpkl files of attributes labeled in image coordinates
        save_path - None or str - if str, specifies full_path to cached homography object, and no other files are required
        default - str static or dynamic or reference
        downsample 
        
        space_dir - 
        im_dir    
        """
        
        # intialize correspondence
        self.downsample = downsample 
        self.polarity = 1
        self.MM_offset = 0
        self.save_file = save_path
        self.default = default
        self.hg_start_time = 0
        self.hg_sec        = 10
        
        self.correspondence = {}
        if save_path is not None and os.path.exists(save_path):
            try:
                with open(save_path,"rb") as f:
                    # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
                    self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets,self.hg_sec,self.hg_start_time = pickle.load(f)
                    
            except:
                with open(save_path,"rb") as f:
                    # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
                    self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets = pickle.load(f)
                    
            # reload  parameters of curvilinear axis spline
            # rather than the spline itself for better pickle reloading compatibility
                
        
        elif aerial_ref_dir is None:
            raise IOError("aerial_im_dir must be specified unless save_path is given")
        
        else:
            # fit roadway coordinate spline

            self.median_tck = None
            self.median_u = None
            self.guess_tck = None
            self.guess_tck2 = None
            self.all_splines = None
            self.yellow_offsets = None
            self._fit_spline(aerial_ref_dir)
            self.save(save_path)
            
        if im_ref_dir is not None:
            try:
                self.load_correspondences(im_ref_dir)
            except:
                aerial_file = os.path.join(aerial_ref_dir,"stateplane_all_points.cpkl")
                for file in os.listdir(im_ref_dir):
                    if ".cpkl" not in file: continue
                    path = os.path.join(im_ref_dir,file)
                    self.generate_reference(aerial_file, path)
                    
        # object class info doesn't really belong in homography but it's unclear
        # where else it should go, and this avoids having to pass it around 
        # for use in height estimation
        if True:
            self.class_dims = {
                    "sedan":[16,6,4],
                    "midsize":[18,6.5,5],
                    "van":[20,6,6.5],
                    "pickup":[20,6,5],
                    "semi":[55,9,14],
                    "truck (other)":[25,9,14],
                    "truck": [25,9,14],
                    "motorcycle":[7,3,4],
                    "trailer":[16,7,3],
                    "other":[18,6.5,5]
                }
            
            self.class_heights = dict([(key,self.class_dims[key][2]) for key in self.class_dims.keys()])
            
            self.class_dict = { "sedan":0,
                        "midsize":1,
                        "van":2,
                        "pickup":3,
                        "semi":4,
                        "truck (other)":5,
                        "truck": 5,
                        "motorcycle":6,
                        "trailer":7,
                        0:"sedan",
                        1:"midsize",
                        2:"van",
                        3:"pickup",
                        4:"semi",
                        5:"truck (other)",
                        6:"motorcycle",
                        7:"trailer"
                        }
        
        
    def save(self,save_file):
        with open(save_file,"wb") as f:
            pickle.dump([self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets,self.hg_sec,self.hg_start_time],f)
        
      
    # def load_correspondence_old(self,im_ref_dir):
    #     """
    #     im_ref_dir - directory of directories of pickle files, each pickle file is a dictionary corresponding to and is loaded into self.correspondence
    #     """
        
    #     # dirs = os.listdir(im_ref_dir)
    #     # for subdir in dirs:
    #     #     if not os.path.isdir(os.path.join(im_ref_dir,subdir)):
    #     #         continue
            
    #     files = glob.glob(os.path.join(im_ref_dir, '*.cpkl'),recursive = True)
        

    #     for file in files:
    #         # match full name (group(x) gives you the parts inside the parentheses )
    #         camera_name = re.match('.*/(P\d\dC\d\d)\.cpkl', file).group(1)

    #         fp = os.path.join(im_ref_dir,file)
            
    #         # parse out relevant data from EB and WB sides
    #         with open(fp,"rb") as f:
    #             data = pickle.load(f)
    #             for side in ["EB","WB"]:
    #                 side_data = data[side]
                
    #                 corr = {}
                    
    #                 corr["P_reference"] = torch.from_numpy(side_data["P"])
    #                 corr["H_reference"] = torch.from_numpy(side_data["H"])

    #                 corr["FOV"] = side_data["FOV"]
    #                 corr["mask"] = side_data["mask"]
                    
    #                 corr_name = "{}_{}".format(camera_name,side)
    #                 self.correspondence[corr_name] = corr
                            
    #     self.hg_sec = 1
    #     self.hg_start_time = 0
            
    #     if False: #temporary passthrough
    #         self.load_correspondences_WACV(im_ref_dir)
            
            
    def load_correspondences(self,im_ref_dir):
        """
        im_ref_dir - directory of directories of pickle files, each pickle file is a dictionary corresponding to and is loaded into self.correspondence
        """
        
        # dirs = os.listdir(im_ref_dir)
        # for subdir in dirs:
        #     if not os.path.isdir(os.path.join(im_ref_dir,subdir)):
        #         continue
            
        files = glob.glob(os.path.join(im_ref_dir, 'hg_*.cpkl'),recursive = True)
        

        for file in files:
            # match full name (group(x) gives you the parts inside the parentheses )
            camera_name = re.match('.*/hg_(P\d\dC\d\d)\.cpkl', file).group(1)

            fp = os.path.join(im_ref_dir,file)
            
            # parse out relevant data from EB and WB sides
            with open(fp,"rb") as f:
                data = pickle.load(f)
                for side in ["EB","WB"]:
                    side_data = data[side]
                    if np.isnan(side_data["HR"].sum()):
                        continue
                    else:
                        corr = {}
                        corr["P_static"] = torch.from_numpy(side_data["PA"])
                        corr["H_static"] = torch.from_numpy(side_data["HA"])
                        corr["P_reference"] = torch.from_numpy(side_data["PR"])
                        corr["H_reference"] = torch.from_numpy(side_data["HR"])
                        corr["P_dynamic"]   = torch.from_numpy(side_data["P"])
                        corr["H_dynamic"]   = torch.from_numpy(side_data["H"])
                        corr["FOV"] = side_data["FOV"]
                        corr["mask"] = side_data["mask"]
                        corr["time"] = side_data["time"]
                        
                        corr_name = "{}_{}".format(camera_name,side)
                        self.correspondence[corr_name] = corr
                            
        self.hg_sec = side_data["time"][1] - side_data["time"][0]
        self.hg_start_time = side_data["time"][0]            
            
        if False: #temporary passthrough
            self.load_correspondences_WACV(im_ref_dir)
    
        
    def load_correspondences_WACV(self,im_ref_dir):
        
        """
        For now, we'll load up the data from the WACV 1 hour save pickle and add the dynamic and static homographies to it
        Then, generate a time method for indexing
        """
        save_path = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/CIRCLES_20_Wednesday_1hour.cpkl"
        with open(save_path,"rb") as f:
            # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
            self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets = pickle.load(f)
        
        removals = []
        # get all files in 
        for corr in self.correspondence:
            print(corr)
            
            if "old" in corr:
                removals.append(corr)
                continue
            # remove old data
            self.correspondence[corr].pop("P",None)
            self.correspondence[corr].pop("H",None)
            self.correspondence[corr].pop("H_inv",None)
            
            # load static P and H
            P_path = os.path.join(im_ref_dir,"static","P_{}.npy".format(corr))
            P = torch.from_numpy(np.load(P_path))
            
            H_path = os.path.join(im_ref_dir,"static","H_{}.npy".format(corr))
            H = torch.from_numpy(np.load(H_path))
            
            #P[:,2] *= -1
            self.correspondence[corr]["P_static"] = P
            self.correspondence[corr]["H_static"] = H
            
            if torch.isnan(P.sum() + H.sum()):
                print("No static hg for {}".format(corr))
            
            # load reference P and H
            P_path = os.path.join(im_ref_dir,"reference","P_{}.npy".format(corr))
            P = torch.from_numpy(np.load(P_path))
            
            H_path = os.path.join(im_ref_dir,"reference","H_{}.npy".format(corr))
            H = torch.from_numpy(np.load(H_path))
            
            #P[:,2] *= -1
            self.correspondence[corr]["P_reference"] = P
            self.correspondence[corr]["H_reference"] = H
            
            if torch.isnan(P.sum() + H.sum()):
                print("No reference hg for {}".format(corr))
                removals.append(corr)
            # load dynamic P and H
            P_path = os.path.join(im_ref_dir,"dynamic","P_{}.npy".format(corr))
            P = torch.from_numpy(np.load(P_path))
            
            H_path = os.path.join(im_ref_dir,"dynamic","H_{}.npy".format(corr))
            H = torch.from_numpy(np.load(H_path))
            
            #P[:,2] *= -1
            self.correspondence[corr]["P_dynamic"] = P
            self.correspondence[corr]["H_dynamic"] = H
            
            if torch.isnan(P.sum() + H.sum()):
                print("No dynamic hg for {}".format(corr))
                
            self.correspondence[corr] = copy.deepcopy(self.correspondence[corr])

        # hg sec is the seconds between dynamic homographies and hg_start time is the timestamp for the 0th homography
        self.hg_start_time = 0
        self.hg_sec        = 10
        
        
        for removal in removals:
            self.correspondence.pop(removal,None)
        
    def generate_reference(self,aerial_file,cam_file):
       
        camera = cam_file.split(".cpkl")[0].split("/")[-1]

        # load aerial points
        with open(aerial_file,"rb") as f:
            aer_data = pickle.load(f)
            
        # load cam points
        with open(cam_file,"rb") as f:
            cam_data = pickle.load(f)
            
        
        for direction in ["EB","WB"]:
            try:
                im_pts = []
                aer_pts = []
                names = []
                # get matching set of points
                for point in cam_data[direction]["points"]:
                    key = point[2]
                    if key in aer_data.keys():
                        im_pts.append(point[0:2])
                        aer_pts.append(aer_data[key])
                        names.append(key)
                        
                # stack pts
                im_pts = np.stack(im_pts)
                aer_pts = np.stack(aer_pts)
            
                # compute homography
                cor = {}
                cor["H"],_     = cv2.findHomography(im_pts,aer_pts)
                cor["H_inv"],_ = cv2.findHomography(aer_pts,im_pts)
                vp = cam_data[direction]["z_vp"]
                
                # P is a [3,4] matrix 
                #  column 0 - vanishing point for space x-axis (axis 0) in image coordinates (im_x,im_y,im_scale_factor)
                #  column 1 - vanishing point for space y-axis (axis 1) in image coordinates (im_x,im_y,im_scale_factor)
                #  column 2 - vanishing point for space z-axis (axis 2) in image coordinates (im_x,im_y,im_scale_factor)
                #  column 3 - space origin in image coordinates (im_x,im_y,scale_factor)
                #  columns 0,1 and 3 are identical to the columns of H, 
                #  We simply insert the z-axis column (im_x,im_y,1) as the new column 2
                
                P = np.zeros([3,4])
                P[:,0] = cor["H_inv"][:,0]
                P[:,1] = cor["H_inv"][:,1]
                P[:,3] = cor["H_inv"][:,2] 
                P[:,2] = np.array([vp[0],vp[1],1])  * 10e-09
                cor["P"] = P
                
                
                # fit Z vp
                self._fit_z_vp(cor,cam_data,direction)
            
                
                
                # store correspodence - map into format expected by new rcs class which has dynamic, static and reference (but set non-reference as Nan)                
                
                corr = {}
                corr["P_static"] = torch.from_numpy(cor["P"]) * torch.nan
                corr["H_static"] = torch.from_numpy(cor["H"]) * torch.nan
                corr["P_reference"] = torch.from_numpy(cor["P"])
                corr["H_reference"] = torch.from_numpy(cor["H"])
                corr["P_dynamic"]   = None
                corr["H_dynamic"]   = None
                corr["FOV"] = cam_data[direction]["FOV"]
            
                if len(cam_data["EB"]["mask"]) > 0:
                    corr["mask"] = cam_data["EB"]["mask"]
                else:
                    corr["mask"] = cam_data["WB"]["mask"]
                corr["time"] = None
                
                corr_name = "{}_{}".format(camera,direction)
                self.correspondence[corr_name] = corr
                
                
                
            except:
                pass
            
    def _fit_z_vp(self,cor,im_data,direction):
        
        print("fitting Z coordinate scale")
        
        P_orig = cor["P"].copy()
        
        max_scale = 10000
        granularity = 1e-12
        upper_bound = max_scale
        lower_bound = -max_scale
        
        # create a grid of 100 evenly spaced entries between upper and lower bound
        C_grid = np.linspace(lower_bound,upper_bound,num = 100,dtype = np.float64)
        step_size = C_grid[1] - C_grid[0]
        iteration = 1
        
        while step_size > granularity:
            
            best_error = np.inf
            best_C = None
            # for each value of P, get average reprojection error
            for C in C_grid:
                
                # scale P
                P = P_orig.copy()
                P[:,2] *= C
                
                
                # search for optimal scaling of z-axis row
                vp_lines = im_data[direction]["z_vp_lines"]
                
                # get bottom point (point # 2)
                points = torch.stack([ torch.tensor([vpl[2] for vpl in vp_lines]), 
                                          torch.tensor([vpl[3] for vpl in vp_lines]) ]).transpose(1,0)
                t_points = torch.stack([ torch.tensor([vpl[0] for vpl in vp_lines]), 
                                          torch.tensor([vpl[1] for vpl in vp_lines]) ]).transpose(1,0)
                heights =  torch.tensor([vpl[4] for vpl in vp_lines]).unsqueeze(1)
    
                
                # project to space
                
                d = points.shape[0]
                
                # convert points into size [dm,3]
                points = points.view(-1,2).double()
                points = torch.cat((points,torch.ones([points.shape[0],1],device=points.device).double()),1) # add 3rd row
                
                H = torch.from_numpy(cor["H"]).transpose(0,1).to(points.device)
                new_pts = torch.matmul(points,H)
                    
                # divide each point 0th and 1st column by the 2nd column
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                # drop scale factor column
                new_pts = new_pts[:,:2] 
                
                # reshape to [d,m,2]
                new_pts = new_pts.view(d,2)
                
                # add third column for height
                new_pts_shifted  = torch.cat((new_pts,heights.double()),1)
                
    
                # add fourth column for scale factor
                new_pts_shifted  = torch.cat((new_pts_shifted,torch.ones(heights.shape)),1)
                new_pts_shifted = torch.transpose(new_pts_shifted,0,1).double()
    
                # project to image
                P = torch.from_numpy(P).double().to(points.device)
                
    
                new_pts = torch.matmul(P,new_pts_shifted).transpose(0,1)
                
                # divide each point 0th and 1st column by the 2nd column
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                # drop scale factor column
                new_pts = new_pts[:,:2] 
                
                # reshape to [d,m,2]
                repro_top = new_pts.view(d,-1,2).squeeze()
                
                # get error
                
                error = torch.pow((repro_top - t_points),2).sum(dim = 1).sqrt().mean()
                
                # if this is the best so far, store it
                if error < best_error:
                    best_error = error
                    best_C = C
                    
            
            # define new upper, lower  with width 2*step_size centered on best value
            #print("On loop {}: best C so far: {} avg error {}".format(iteration,best_C,best_error))
            lower_bound = best_C - 2*step_size
            upper_bound = best_C + 2*step_size
            C_grid = np.linspace(lower_bound,upper_bound,num = 100,dtype = np.float64)
            step_size = C_grid[1] - C_grid[0]
    
            #print("New C_grid: {}".format(C_grid.round(4)))
            iteration += 1
        
        
        
        P_new = P_orig.copy()
        P_new[:,2] *= best_C
        cor["P"] = P_new
        
    def _fit_spline(self,space_dir,use_MM_offset = True):
        """
        Spline fitting is done by:
            1. Assemble all points labeled along a yellow line in either direction
            2. Fit a spline to each side of each line
            3. Sample each spline at fine intervals
            4. Use finite difference method to determine the distance along the spline for each fit point
            5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
            5b. For each line, sample and define the set of points midway between
            5c. For each line, define a separate spline fit to these points that defines the position of that lane as a function of x (along roadway)
            6. Sample each yellow-line spline at fine intervals
            7. Move along one spline and at each point, find the closest point on each other spline
            8. Define a point on the median/ midpoint axis as the average on these 4 splines
            9. Use the set of median points to define a new spline
            10. Use the finite difference method to reparameterize this spline according to distance along it
            11. Optionally, compute a median spline distance offset from mile markers
            12. Optionally, recompute the same spline, this time accounting for the MM offset
            
        
            
            space_dir - str - path to directory with csv files of attributes labeled in space coordinates
            use_MM_offset - bool - if True, offset according to I-24 highway mile markers
        """
        
        print("Fitting median spline..")
        
        samples_per_foot = 12
        splines = {}
       
        # First, load all annotations
        for direction in ["EB","WB"]:
            for line_side in ["i","o"]:
                ### State space, do once
                
                ae_x = []
                ae_y = []
                ae_id = []
                
                # 1. Assemble all points labeled along a yellow line in either direction
                for file in os.listdir(space_dir):
                    if direction.lower() not in file:
                        continue
                    
                
                    
                    # load all points
                    dataframe = pd.read_csv(os.path.join(space_dir,file))
                    try:
                        dataframe = dataframe[dataframe['point_pos'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["point_id"].tolist()
                        st_id = [attribute_name + "_" + item for item in feature_idx]
                        
                        st_x = dataframe["x"].tolist()
                        st_y = dataframe["y"].tolist()
                        #st_x = dataframe["st_x"].tolist()
                        #st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                    except:
                        dataframe = dataframe[dataframe['side'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["id"].tolist()
                        side        = dataframe["side"].tolist()
                        st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                        
                        st_x = dataframe["x"].tolist()
                        st_y = dataframe["y"].tolist()
                        #st_x = dataframe["st_x"].tolist()
                        #st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                
                
                for line in ["yel{}".format(line_side), "d1{}".format(line_side),"d2{}".format(line_side),"d3{}".format(line_side)]:
                    #print("On line {} {}".format(line,direction))
                    ae_spl_x = []
                    ae_spl_y = []
                    ae_spl_u = []  # u parameterizes distance along spline 
                    ae_spl_id = []                
                    
                    letter_to_side = {"a":"i",
                                      "b":"i",
                                      "c":"o",
                                      "d":"o"
                                      }
                    
                    for i in range(len(ae_x)):
                        try:
                            if line in ae_id[i] or ( len(ae_id[i].split("_")) == 4  and line == ae_id[i].split("_")[1] + letter_to_side[ae_id[i].split("_")[3]]):
                            #if "yel{}".format(line_side) in ae_id[i]:
                                ae_spl_x.append(ae_x[i])
                                ae_spl_y.append(ae_y[i])
                                ae_spl_id.append(ae_id[i])
                        except KeyError:
                            pass
        
                    # if possible, use spline to smooth points
                    # if self.median_tck is not None:
                    #     ae_spl_x,ae_spl_y,ae_spl_id = self.shift_aerial_points2( ae_spl_x,ae_spl_y,ae_spl_id)
        
                    # 2. Fit a spline to each of EB, WB inside and outside
                     
                    # find a sensible smoothness parameter
                    
                    # compute the yellow line spline in state plane coordinates (sort points by y value since road is mostly north-south)
                    ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                    ae_data,idx = np.unique(ae_data,axis = 1,return_index = True)
                    order = np.argsort(ae_data[0,:])
                    ae_data2 = ae_data[:,order]#[::-1]]
                    order2 = np.argsort(ae_data[0,:])
                    ae_data = ae_data2.copy()
                    # 3. Sample the spline at fine intervals
                    # get spline and sample points on spline
                    w = np.ones(ae_data.shape[1])
                    
                    ae_spl_x  = [ae_spl_x[i]  for i in idx]
                    ae_spl_y  = [ae_spl_y[i]  for i in idx]
                    ae_spl_id = [ae_spl_id[i] for i in idx]
                    # for dim in [0,1]:
                    #     width = 13
                    #     extend1 = torch.ones((width-1)//2) * ae_data[dim,0]
                    #     extend2 = torch.ones((width-1)//2) * ae_data[dim,-1]
                    #     ys_extended = torch.cat([extend1,torch.from_numpy(ae_data[dim,:]),extend2])
                
                    #     smoother = np.hamming(width)
                    #     smoother = smoother/ sum(smoother)
                    #     ys = np.convolve(ys_extended,smoother,mode = "valid")
                    #     ae_data[dim,:] = ys
                    
                    knot_spacing = 0
                    s0 = 0.1
                    try:
                        ae_tck, ae_u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, per=False) 
                    except ValueError as e:
                        print(e)
                   
                    span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                    ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), ae_tck)

                    # TEMP
                    # plt.plot(ae_data[0,:],ae_data[1,:], "o-")
                    # legend.append(direction + "_" + line_side)
        
                    # 4. Use finite difference method to determine the distance along the spline for each fit point
                    fd_dist = np.concatenate(  (np.array([0]),  ((ae_x_prime[1:] - ae_x_prime[:-1])**2 + (ae_y_prime[1:] - ae_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
                    integral_dist = np.cumsum(fd_dist)
                    
                    # for each fit point, find closest point on spline, and assign it the corresponding integral distance
                    for p_idx in range(len(ae_spl_x)):
                        # I think these should instead reference ae_data because that one has been sorted
                        #px = ae_spl_x[p_idx]
                        #py = ae_spl_y[p_idx]
                        px = ae_data[0,p_idx]
                        py = ae_data[1,p_idx]
                        
                        dist = ((ae_x_prime - px)**2 + (ae_y_prime - py)**2)**0.5
                        min_dist,min_idx= np.min(dist),np.argmin(dist)
                        ae_spl_u.append(integral_dist[min_idx])
                    
                    # 5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
                    #ae_spl_u.reverse()
                    
                    # sort by increasing u
                    ae_spl_u = np.array(ae_spl_u)
                    sorted_idxs = np.argsort(ae_spl_u)
                    ae_spl_u = ae_spl_u[sorted_idxs]
                    ae_data  = ae_data[:,sorted_idxs]
                    
                    while True and  knot_spacing < 10:
                        s0 *= 1.2
                        tck, u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, u = ae_spl_u)
                        knots = tck[0]
                        knot_spacing = np.min(np.abs(knots[5:-4] - knots[4:-5]))
                        #print(knot_spacing,s0)
                    tck, u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, u = ae_spl_u)
                    splines["{}_{}_{}".format(line,direction,line_side)] = [tck,u]
                

            
        # to prevent any bleedover
        del dist, min_dist, min_idx, ae_spl_y,ae_spl_x, ae_spl_u, ae_data
           
        import matplotlib.pyplot as plt
        plt.figure()
        legend = []
        # 6. Sample each of the 4 splines at fine intervals 
        for key in splines:
            tck,u = splines[key]
    
            span_dist = np.abs(u[0] - u[-1])
            x_prime, y_prime = interpolate.splev(np.linspace(u[0], u[-1], int(span_dist)), tck)
            splines[key].append(x_prime)
            splines[key].append(y_prime)
            
            plt.plot(x_prime,y_prime)
            legend.append(key)
        
        ###### Now, for each pair of splines, sample each and get a midway spline. These will be the splines we use
        
        for direction in ["EB","WB"]:
            for line in ["yel","d1","d2","d3"]:
                new_key = "{}_{}_center".format(direction,line)
                print("Getting smooth centered spline for {}".format(new_key))
                
                for key in splines.keys():
                    if direction in key and line in key and "i" in key:
                        i_spline = splines[key][0] # just tck
                    elif direction in key and line in key and "o" in key:
                        o_spline = splines[key][0] # just tck
                        u_range =  np.linspace(np.min(splines[key][1]), np.max(splines[key][1]), 50)
                
                # sample each spline at fine interval 
                #u_range = np.array(med_spl_u )
                
                
                # sample each spline at the same points
                x_in,y_in  = np.array(interpolate.splev(u_range,i_spline))
                x_out, y_out = np.array(interpolate.splev(u_range,o_spline))
                
                # average the two points
                y_mid = (y_in + y_out)/2
                x_mid = (x_in + x_out )/2
                
                data = np.stack([x_mid,y_mid])
                # fit spline y(u)
                s0 = 0.1
                knot_spacing = 0
                while knot_spacing < 100: # adjust spline smoothness for each center line
                    s0 *= 1.25
                    mid_line_tck,mid_line_u = interpolate.splprep(data, s=s0, u = u_range)
                    knots = mid_line_tck[0]
                    knot_spacing = np.min(np.abs(knots[5:-4] - knots[4:-5]))
                    #print(knot_spacing,s0)
                # store
            
                splines[new_key] = [mid_line_tck,mid_line_u]
                
                # plot
                x_prime, y_prime = interpolate.splev(np.linspace(u_range[0], u_range[-1], 5000), mid_line_tck)
                splines[new_key].append(x_prime)
                splines[new_key].append(y_prime)
                plt.plot(x_prime,y_prime)
                legend.append(new_key)

        # plt.legend(legend)
        # plt.show()        

        # # cache spline for each lane for plotting purposes
        # self.all_splines = splines 
        # return

        med_spl_x = []
        med_spl_y = []
        
        
        
        print("sampling yellow line splines")
        
        # 7. Move along one spline and at each point, find the closest point on each other spline
        # by default, we'll use EB_o as the base spline
        for main_key in ["yelo_EB_o","yeli_EB_i","yelo_WB_o","yeli_WB_i"]:
        #for main_key in ["yelo_WB_o","yeli_WB_i"]:

            main_spl = splines[main_key]
            main_x = main_spl[2]
            main_y = main_spl[3]
            
            
            for p_idx in range(len(main_x)):
                px,py = main_x[p_idx],main_y[p_idx]
                points_to_average = [np.array([px,py])]
                
                for key in splines:
                    if key != main_key:
                        if key not in ["yelo_WB_o","yeli_WB_i","yelo_EB_o","yeli_EB_i"]: continue
                        arr_x,arr_y = splines[key][2], splines[key][3]
                        
                        dist = np.sqrt((arr_x - px)**2 + (arr_y - py)**2)
                        min_dist,min_idx= np.min(dist),np.argmin(dist)
                        
                        points_to_average.append( np.array([arr_x[min_idx],arr_y[min_idx]]))
                
                if len(points_to_average) < 4:
                    print("Outlier removed")
                    continue
                
                med_point = sum(points_to_average)/len(points_to_average)
                
                
                
                # 8. Define a point on the median/ midpoint axis as the average on these 4 splines
                med_spl_x.append(med_point[0])
                med_spl_y.append(med_point[1])
            
        print("Done sampling")
        
        # 9. Use the set of median points to define a new spline
        # sort by increasing x
        med_data = np.stack([np.array(med_spl_x),np.array(med_spl_y)])
        med_data = med_data[:,np.argsort(med_data[0])]
        
        # remove weirdness (i.e. outlying points) s.t. both x and y are monotonic and strictly increasing
        keep = (med_data[1,1:] <  med_data[1,:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        
        keep = (med_data[0,1:] >  med_data[0,:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        #med_data = np.ascontiguousarray(med_data)
        

        
        s = 10
        n_knots = len(med_data[0])
        while n_knots > 300:
            med_tck,med_u = interpolate.splprep(med_data, s=s, per=False)
            n_knots = len(med_tck[0])
            s = s**1.2
            print("Fitting median spline, n_knots = {}".format(n_knots))
        
        # 10. Use the finite difference method to reparameterize this spline according to distance along it
        med_spl_x = med_data[0]
        med_spl_y = med_data[1]
        span_dist = np.sqrt((med_spl_x[0] - med_spl_x[-1])**2 + (med_spl_y[0] - med_spl_y[-1])**2)
        med_x_prime, med_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), med_tck)
        
        
        med_fd_dist = np.concatenate(  (np.array([0]),  ((med_x_prime[1:] - med_x_prime[:-1])**2 + (med_y_prime[1:] - med_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
        med_integral_dist = np.cumsum(med_fd_dist)
        
        # for each fit point, find closest point on spline, and assign it the corresponding integral distance
        med_spl_u = []
        print("Getting integral distance along median spline")
        for p_idx in range(len(med_data[0])):
            px,py = med_data[0,p_idx], med_data[1,p_idx]
            
            dist = ((med_x_prime - px)**2 + (med_y_prime - py)**2)**0.5
            min_dist,min_idx= np.min(dist),np.argmin(dist)
            med_spl_u.append(med_integral_dist[min_idx])
        
        
        # sort by increasing u I guess
        med_spl_u = np.array(med_spl_u)
        sorted_idxs = np.argsort(med_spl_u)
        med_spl_u = med_spl_u[sorted_idxs]
        med_data  = med_data[:,sorted_idxs]
        
        # sort by strictly increasing u
        keep = (med_spl_u[1:] >  med_spl_u[:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        med_spl_u = med_spl_u[keep.nonzero()[0]]
        
        import matplotlib.pyplot as plt
        #plt.figure(figsize = (20,20))
        plt.plot(med_data[0],med_data[1])
        legend.append("Median")
        
       
        
        
        smoothing_dist = 500
        max_allowable_dev = 1
        # at this point, we have the median data and the integrated median distances (med_spl_u) and med_data
        # Let's try simply finding a single spline with <smoothing_dist> spaced fit-points and high-weighted edges
        
        
        
  
        
        
        # plt.figure()
        # plt.plot(med_spl_u)
        # plt.plot(np.array(med_spl_u)[np.argsort(med_spl_u)])
        # plt.legend(["Unsorted","Sorted"])
        
        
        s = 8
        min_dist = 0
        max_dev = 0

        while min_dist < smoothing_dist and max_dev < max_allowable_dev:
            final_tck,final_u = interpolate.splprep(med_data.astype(float), s=s, u=med_spl_u)
            knots = final_tck[0]
            min_dist = np.min(np.abs(knots[4:] - knots[:-4]))
            
            current_x,current_y = interpolate.splev(med_spl_u,final_tck)
            
            dist = ((current_x - med_data[0,:])**2 + (current_y - med_data[1,:])**2)**0.5
            max_dev = np.max(dist)
            
            print("With s = {}, {} knots, and min knot spacing {}, max spline - median point deviation = {}".format(s,len(knots),min_dist,max_dev))
            s = s**1.1
        
        #final_tck, final_u = interpolate.splprep(med_data, u = med_spl_u)
        self.median_tck = final_tck
        self.median_u = final_u
        
        
        # sample for final plotting
        final_plot_x,final_plot_y = interpolate.splev(np.linspace(min(med_spl_u), max(med_spl_u), 2000), final_tck)
        plt.plot(final_plot_x,final_plot_y)
        legend.append("Final Spline")
        
        
    
        
        
        # cache spline for each lane for plotting purposes
        self.all_splines = splines 

        
        
        
        
        
        ### get the inverse spline g(x) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        print(med_data.shape,med_spl_u.shape)
    
    
        # get guess_tck from all sparse 
        if True:
            med_data = np.array([final_plot_x,final_plot_y])
            med_spl_u = np.linspace(min(med_spl_u), max(med_spl_u), 2000)

            
        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[0])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]
    
        self.guess_tck = interpolate.splrep(med_data[0].astype(float),med_spl_u.astype(float))
        
        # get the secondary inverse spline g(y) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        
        # a second copy for later which won't be resorted
        u_range = np.array(med_spl_u )
        
        print(med_data.shape,med_spl_u.shape)
    
    
        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[1])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]
        self.guess_tck2 = interpolate.splrep(med_data[1].astype(float),med_spl_u.astype(float))
            

        plt.legend(legend)
        plt.title("Individual splines")
        plt.show()    
            
        
        ### Offset by MM
        # if use_MM_offset:
        #     # 11. Optionally, compute a median spline distance offset from mile markers
        #     self.MM_offset = self._fit_MM_offset(space_dir)
        
        
        ### get y(u) splines for eastbound and westbound side yellow lines
        
        if True:
            self.yellow_splines = {}
            
            for direction in ["EB", "WB"]:
                i_spline = splines["yeli_{}_i".format(direction)][0] # just tck
                
            
                # sample each spline at fine interval 
                u_range = np.array(med_spl_u )
                sorted_idxs = np.argsort(u_range)
                u_range = u_range[sorted_idxs]
                
                # sample each spline at the same points
                y_in  = np.array(interpolate.splev(u_range,i_spline))
                med   = np.array(interpolate.splev(u_range,self.median_tck))
                
                dist = np.sum(((y_in-med)**2),axis = 0)**0.5
                if direction == "WB": dist *= -1
        
                self.yellow_splines[direction] = interpolate.splrep(u_range,dist,s = 1)
                self._generate_lane_offset(aerial_ref_dir,)                

            
            
    
    
    
    def closest_spline_point(self,points, epsilon = 0.01, max_iterations = 10):
        """
        Given a tensor of points in 3D space, find the closest point on the median spline
        for each point as follows:
            1. Query self.guess_tck spline to get f(x) = u initial guess
            2. Use Newton's method to find the point where dist = min
      
        points         - [d,3] tensor of points in state plane coordinates
        epsilon        - float - keep iterating while max change > epsilon or ii.
        max_iterations - int - ii. keep iterating until n_iterations = max_iterations
        RETURNS:         [d] tensor of coordinates along spline axis
        """
        PLOT = False
        leg = []
        start = time.time()
        # intial guess at closest u values
        points = points.cpu().data.numpy()
        #points = points[:1,:]
        guess_u = interpolate.splev(points[:,0],self.guess_tck)
        guess_u2 = interpolate.splev(points[:,1],self.guess_tck2)
        guess_u = (guess_u + guess_u2)/2.0
        
        #guess_u *= 0
        
        
        
        # guesses = np.linspace(guess_u-500,guess_u+500,103)
        # guess_x,guess_y = interpolate.splev(guesses, self.median_tck)
        
        # plt.scatter(guess_x[:,0],guess_y[:,0])
        # leg.append("Initial samples")
        
        # dist = np.sqrt((guess_x - points[:,0][np.newaxis,:].repeat(guess_x.shape[0],axis = 0))**2 + (guess_y - points[:,1][np.newaxis,:].repeat(guess_y.shape[0],axis = 0))**2)
        # min_idx = np.argmin(dist,axis = 0)
        
        # guess = guesses[min_idx,[i for i in range(guesses.shape[1])]]
        # guesses = np.linspace(guess-10,guess+10,200)
        # guess_x,guess_y = interpolate.splev(guesses, self.median_tck)
        
        # dist = np.sqrt((guess_x - points[:,0])**2 + (guess_y - points[:,1])**2)
        # min_idx = np.argmin(dist,axis = 0)
        
        #guess_u = guesses[min_idx,[i for i in range(guesses.shape[1])]]
        
        
        it = 0
        max_change = np.inf
        while it < max_iterations and max_change > epsilon:
            spl_x,spl_y             = interpolate.splev(guess_u,self.median_tck)
            spl_xx,spl_yy = interpolate.splev(guess_u,self.median_tck, der = 1)
            spl_xxx,spl_yyy = interpolate.splev(guess_u,self.median_tck, der = 2)

            
            dist_proxy = (spl_x - points[:,0])**2 + (spl_y - points[:,1])**2
            dist_proxy_deriv = (spl_x-points[:,0])*spl_xx + (spl_y-points[:,1])*spl_yy
            #dist_proxy_deriv2 = (2*spl_xx**2)+2*(spl_x-points[:,0])*spl_xxx + (2*spl_yy**2)+2*(spl_y-points[:,1])*spl_yyy
            dist_proxy_deriv2 = (spl_xx**2)+(spl_x-points[:,0])*spl_xxx + (spl_yy**2)+(spl_y-points[:,1])*spl_yyy

            
            new_u = guess_u - dist_proxy_deriv/dist_proxy_deriv2
            
            max_change = np.max(np.abs(new_u-guess_u))
            it += 1
            
            guess_u = new_u
            
            if PLOT:
                plt.scatter(spl_x[:1],spl_y[:1])
                leg.append(it)
                
            #print("Max step: {}".format(max_change))
         
        #print("Newton method took {}s for {} points".format(time.time() - start,points.shape[0]))
        
        if PLOT:
            
            # plt.scatter(guess_x[min_idx[0],0],guess_y[min_idx[0],0])
            # leg.append("Closest")
            
            plt.scatter(points[0,0],points[0,1])
            leg.append("Point")
            
            splx,sply = interpolate.splev(np.linspace(0,30000,100000),self.median_tck)
            plt.plot(splx,sply)
            leg.append("Median Spline")
            
            plt.legend(leg)
            plt.axis("equal")
            plt.show()
            raise Exception
        
        return guess_u
            
    
    
        
    def _generate_lane_offset(self,space_dir,SPLINE_OFFSET = False,SHIFT = False):
        """
        0.) Load all points into line_pts
        1.) Get mean for yellow line on each side
        2.) For each point, subtract yellow line and add back in mean yellow line
        3.) Find best fit line (mean/ std) for each line
        4.) Plot each
        5.) Save lane offsets?
        """
    
        def safe(x):
            try:
                return x.item()
            except:
                return x
            
        # 0. assemble all lane marking coordinates across all correspondences

        line_pts = {}
        
        for direction in ["eb","wb"]:
            ### State space, do once
            

            
            # 1. Assemble all points labeled along a yellow line in either direction
            for file in os.listdir(space_dir):
                if direction.lower() not in file or ("yel" not in file and "d1" not in file and "d2" not in file and "d3" not in file and "d4" not in file):
                    continue
                
                print(file)
                
                
                ae_x = []
                ae_y = []
                ae_id = []
                # load all points
                dataframe = pd.read_csv(os.path.join(space_dir,file))
                try:
                    dataframe = dataframe[dataframe['point_pos'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["point_id"].tolist()
                    st_id = [attribute_name + "_" + item for item in feature_idx]
                    
                    st_x = dataframe["x"].tolist()
                    st_y = dataframe["y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
                except:
                    dataframe = dataframe[dataframe['side'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["id"].tolist()
                    side        = dataframe["side"].tolist()
                    st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                    
                    st_x = dataframe["x"].tolist()
                    st_y = dataframe["y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
            
                # if direction == "wb":
                if SHIFT:
                    ae_x,ae_y,ae_id = self.shift_aerial_points2(ae_x,ae_y,ae_id)    
            
                    
            
                    if False:
                        # save updated spline_points here
                        if "yel" not in file:
                            id_trunc = [item.split("_")[-2] + "_" + item.split("_")[-1] for item in ae_id]
                        else:
                            id_trunc = [int(item.split("_")[-1]) for item in ae_id]
                            
                        update_dict = dict([(id_trunc[i], [safe(ae_x[i]),safe(ae_y[i])]) for i in range(len(ae_id))])     
                        orig_ids = feature_idx
                        
                        x_update = []
                        y_update = []
                        for item in orig_ids:
                            new_item = update_dict[item]
                            x_update.append(new_item[0])
                            y_update.append(new_item[1])
                            
                        dataframe["x"] = x_update
                        dataframe["y"] = y_update
                        dataframe.to_csv(os.path.join("shifted_aerial_points",file))    
            
                try:
                    line_pts[attribute_name][0].append(ae_x)
                    line_pts[attribute_name][1].append(ae_y)
                except KeyError:
                    line_pts[attribute_name] = [ae_x,ae_y]
                    
        if self.yellow_offsets is None:
            yellow_offsets = {}
            for direction in ["EB","WB"]:
                # 1. Get mean yellow line value
                spl = self.all_splines["yeli_{}_i".format(direction.upper())]
                ran = max(spl[1]) - min(spl[1])
                
                u = np.linspace(min(spl[1]),max(spl[1]),int(ran))
                
               
                tck = spl[0]
                yelx,yely = interpolate.splev(u,tck)
                yel_space = torch.tensor([yelx,yely,torch.zeros(len(yelx))]).permute(1,0).unsqueeze(1)
                yel_state = self.space_to_state(yel_space)
                import matplotlib.pyplot as plt

                plt.plot(yel_state[:,0],yel_state[:,1])
                plt.show()
                
                ys  = yel_state[:,1]
                
                # extend and smooth y's
                width = 1205
                extend1 = torch.ones((width-1)//2) * ys[0]
                extend2 = torch.ones((width-1)//2) * ys[-1]
                ys_extended = torch.cat([extend1,ys,extend2])
    
                smoother = np.hamming(width)
                smoother = smoother/ sum(smoother)
                ys = np.convolve(ys_extended,smoother,mode = "valid")
                
                
                
                bin_width = 10
                offsets = np.zeros(10000)
                counts = np.zeros(10000)
                
                for idx in range(len(u)):
                    binidx = int(u[idx]//bin_width)
                    offsets[binidx] += ys[idx]
                    counts[binidx]  += 1
                
                offsets = offsets / (counts + 1e-04)
                
                idx = 0
                while offsets[idx] == 0:
                    idx += 1
                offsets[:idx] = offsets[idx]
                
                idx = -1
                while offsets[idx] == 0:
                    idx-= 1
                    offsets[idx:] = offsets[idx]
                    
                yellow_offsets[direction] = offsets
            
            
            self.yellow_offsets = yellow_offsets
            self.save(self.save_file)
        
        # 2. for each lane, convert all points to roadway coordinates
        
        roadway_data = {}
        for key in line_pts.keys():
            
            if "eb" in key or "EB" in key:
                ymean = 12
                direction = "EB"
            else:
                ymean = -12
                direction = "WB"
                
                
            data = line_pts[key]
            space_data = torch.tensor([data[0],data[1],torch.zeros(len(data[0]))]).permute(1,0).unsqueeze(1)
            
            road_data = self.space_to_state(space_data)
            order = torch.argsort(road_data[:,0])
            
            
            #tck = self.all_splines[direction.upper()+ "_yel_center"][0]
            
            # TODO - let's use the black lines we plot as the yellow offset spline since they are obviously right
            # if direction == "WB":
            #     road_data[:,1] = road_data[:,1] - yel_state[:,1] + ymean
            # else:
            if SPLINE_OFFSET:
                
                # tck = self.all_splines["yeli_{}_i".format(direction.upper())][0]
                # yelx,yely = interpolate.splev(road_data[:,0],tck)
                # yel_space = torch.tensor([yelx,yely,torch.zeros(len(yelx))]).permute(1,0).unsqueeze(1)
                # yel_state = self.space_to_state(yel_space)
                #                                         # TODO - let's use the black lines we plot as the yellow offset spline since they are obviously right
                # road_data[:,1] = road_data[:,1] - yel_state[:,1] + ymean
                
                # new offset style
                road_data_offset_bins = (road_data[:,0] /10).int()
                offsets = yellow_offsets[direction][road_data_offset_bins]
                road_data[:,1] = road_data[:,1] - offsets + ymean
                
            roadway_data[key] = road_data[order,:2]
            
        # 3. Get mean and stddev for line
        for key in roadway_data.keys():
            mean = torch.mean(roadway_data[key][:,1])
            std =  torch.std(roadway_data[key][:,1])
            print("Stats for {}: mean {}, stddev {}".format(key,mean,std))
        
        
        
        
        
        import matplotlib.pyplot as plt
        plt.figure(figsize = (4,4))
        leg= []
        leg2 = []
        # 5. Plot
        
        for key in roadway_data.keys():
            if "yel" in key:
                color = (1,1,0)
            else:
                color = (0,0,0)
            legentry1 = plt.plot(roadway_data[key][:,0],roadway_data[key][:,1],color = color)
            leg.append(key)
        
        if False and self.all_splines:
            for key in self.all_splines.keys():
                if True and "center" not in key:
                    
                    if "eb" in key or "EB" in key:
                        ymean = 12
                        direction = "EB"
                    else:
                        ymean = -12
                        direction = "WB"
                    
                    spline = self.all_splines[key]
                    xs,ys = spline[2],spline[3]
                    
                    space_data = torch.tensor([xs,ys,torch.zeros(len(xs))]).permute(1,0).unsqueeze(1)
                    state_data = self.space_to_state(space_data)
                    
                    #tck = self.all_splines[direction.upper()+ "_yel_center"][0]
                    
                    if SPLINE_OFFSET:
                        # tck = self.all_splines["yeli_{}_i".format(direction.upper())][0]
                        # yelx,yely = interpolate.splev(state_data[:,0],tck)
                        # yel_space = torch.tensor([yelx,yely,torch.zeros(len(yelx))]).permute(1,0).unsqueeze(1)
                        # yel_state = self.space_to_state(yel_space)
                        # state_data[:,1] = state_data[:,1] - yel_state[:,1] + ymean
                        
                        # new offset style
                        road_data_offset_bins = (state_data[:,0]/10).int()
                        offsets = yellow_offsets[direction][road_data_offset_bins]
                        state_data[:,1] = state_data[:,1] - offsets + ymean
                    
                    
                    xs,ys = state_data[:,0],state_data[:,1]
                    
                    # try smoothing
                    if True:
                        width = 1205
                        extend1 = torch.ones((width-1)//2) * ys[0]
                        extend2 = torch.ones((width-1)//2) * ys[-1]
                        ys_extended = torch.cat([extend1,ys,extend2])
    
                        smoother = np.hamming(1205)
                        smoother = smoother/ sum(smoother)
                        ys = np.convolve(ys_extended,smoother,mode = "valid")
                        
                        
                    #xs,ys = interpolate.splev(np.linspace(0,25000,1000),tck)
    
                    # sample at 1000 points
                    plt.plot(xs,ys,color = (0,0,0))
                    leg.append(key)
        
        # if True:
        #     # for each correspondence, plot all of the image points in roadway coordinates
        #     for corr in self.correspondence:
        #         direction = corr.split("_")[-1]
        #         if direction == "WB":  ymean = -12
        #         else: ymean = 12
        #         im_pts = self.correspondence[corr]["corr_pts"]
        #         im_pts = torch.from_numpy(im_pts).unsqueeze(1)
        #         rcs_pts = self.im_to_state(im_pts, heights = torch.zeros(im_pts.shape[0]),name = [corr for _ in range(im_pts.shape[0])],refine_heights = False)
                
        #         if SPLINE_OFFSET:
        #             road_data_offset_bins = (rcs_pts[:,0]/10).int()
        #             offsets = yellow_offsets[direction][road_data_offset_bins]
        #             rcs_pts[:,1] = rcs_pts[:,1] - offsets + ymean
                    
        #         legentry2 = plt.scatter(rcs_pts[:,0],rcs_pts[:,1], color = (0,0,1), marker = ".")
        
        # plt.xlabel("Roadway X (ft)")
        # plt.ylabel("Roadway Y (ft)")
        # plt.legend([legentry2],["Image Correspondence Points, Transformed"])

        
    
    
    def time_corr(self,name,times= None,mode = "H"):
        """
        Returns correspondence per name according to:
            1. time-varying (dynamic) homography at a given time if time is specified
            2. best-fit average homography 
            3. original reference homography
                
        name - str or str list [d]    - correspondence keys in self.correspondence
        times - None or float or float tensor [d] - corresponding times in unix timestamp seconds
        """
        assert mode in ["P","H"] , "Invalid mode in time_corr: {}".format(mode)
        
        
        #print("Need to verify behavior of time_corr")
        
        # deal with single times case - expand into tensor
        if type(times) == float and type(name) == list:
            times = torch.zeros([len(name)]) + times
        
        # if name is a single string, we'll return a single time 
        if type(name) == str:
            if type(times) == torch.Tensor:
                print("Warning: in time_corr(), a single correspondence was specified with multiple times. A single time will be used")
            
            if times is not None:
                # get time index
                tidx = int(min(max(0,int((times[0] - self.hg_start_time) // self.hg_sec)),len(self.correspondence[name]["H_dynamic"])-1))
            
            if mode == "H":
                if times is not None:
                    mat = self.correspondence[name]["H_dynamic"][tidx]
                if times is None or torch.isnan(mat.sum()) or self.default == "static" or self.default == "reference":
                    mat = self.correspondence[name]["H_static"]
                    if torch.isnan(mat.sum()) or self.default == "reference":
                        mat = self.correspondence[name]["H_reference"]
            elif mode == "P":
                if times is not None:
                    mat = self.correspondence[name]["P_dynamic"][tidx]
                if times is None or torch.isnan(mat.sum()) or self.default == "static" or self.default == "reference":
                    mat = self.correspondence[name]["P_static"]
                    if torch.isnan(mat.sum()) or self.default == "reference":
                        mat = self.correspondence[name]["P_reference"]
            return mat
        
        # deal with a list of correspondences
        else:
            # get time index
            if times is not None:
                tidx = ((times - self.hg_start_time) // self.hg_sec)
            
            if mode == "H":
                if times is not None:
                    mat = torch.from_numpy(np.stack([self.correspondence[name[n]]["H_dynamic"][int(min(max(0,tidx[n]),len(self.correspondence[name[n]]["H_dynamic"])-1))] for n in range(len(name))])) 
                else:
                    mat = torch.zeros([len(name),3,3]) * torch.nan
                    
                for m in range(mat.shape[0]): #mat = [d,3,3] tensor, inspect each H matrix
                    if torch.isnan(mat[m].sum()) or self.default == "static" or self.default == "reference":
                        mat[m] = self.correspondence[name[m]]["H_static"]
                        if torch.isnan(mat[m].sum()) or self.default == "reference":
                            mat[m] = self.correspondence[name[m]]["H_reference"]
            elif mode == "P":
                if times is not None:
                    mat = torch.from_numpy(np.stack([self.correspondence[name[n]]["P_dynamic"][int(min(max(0,tidx[n]),len(self.correspondence[name[n]]["P_dynamic"])-1))] for n in range(len(name))])) 
                else:
                    mat = torch.zeros([len(name),3,4]) * torch.nan
                for m in range(mat.shape[0]): #mat = [d,3,3] tensor, inspect each H matrix
                    if torch.isnan(mat[m].sum()) or self.default == "static" or self.default == "reference":
                        mat[m] = self.correspondence[name[m]]["P_static"]
                        if torch.isnan(mat[m].sum()) or self.default == "reference":
                            mat[m] = self.correspondence[name[m]]["P_reference"]
                            

            return mat
        
    # Convenience aliases for time_corr
    def get_H(self,name,times):
        return self.time_corr(name,times,mode = "H")
    def get_P(self,name,times):
        return self.time_corr(name,times,mode = "P")
    
    #%% Conversion Functions
    @safe_name
    def _im_sp(self,points,
               heights = None, 
               name = None, 
               times = None,
               refine_heights = False):
        """
        Converts points by means of perspective transform from image to space
        points    - [d,m,2] array of points in image
        name      - list of correspondence key names
        heights   - [d] tensor of object (guessed) heights or 0
        times     - None or float or [float] - time or times for positions
        refine_heights - bool - if True, points are reprojected back into image and used to rescale the heights
        RETURN:     [d,m,3] array of points in space 
        """
        
        # deal with no points case
        d = points.shape[0]
        if d == 0:
            return torch.empty(0,8,3)
        
        # deal with single name case - expand into list
        if type(name) == list and len(name[0].split("_")) == 1:
            temp_name = [sub_n+ "_WB" for sub_n in name]
            name = temp_name
        
        
            
        
        n_pts = points.shape[1]
        # convert points into size [dm,3]
        points = points.view(-1,2).double()
        points = torch.cat((points,torch.ones([points.shape[0],1],device=points.device).double()),1) # add 3rd row
        
        
        
        ### TODO NOTE - suspect the transpose to cause problems here
        if heights is not None:
            
            if type(name) == list:
                H = self.get_H(name,times).transpose(1,2).double() 
                H = H.unsqueeze(1).repeat(1,n_pts,1,1).view(-1,3,3).to(points.device)
                points = points.unsqueeze(1)
                new_pts = torch.bmm(points,H)
                new_pts = new_pts.squeeze(1)
            else:
                H = self.get_H(name,times).transpose(0,1).to(points.device).double()
                new_pts = torch.matmul(points,H)
            
            # divide each point 0th and 1st column by the 2nd column
            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
            
            # drop scale factor column
            new_pts = new_pts[:,:2] 
            
            # reshape to [d,m,2]
            new_pts = new_pts.view(d,-1,2)
            
            # add third column for height
            new_pts = torch.cat((new_pts,torch.zeros([d,new_pts.shape[1],1],device = points.device).double()),2)
            
            # overwrite last_4 points with first 4 (i.e. top points are gibberish and not meaningful)
            if n_pts == 8:
                new_pts[:,4:,:] = new_pts[:,:4,:]
            
                if type(heights) == torch.Tensor: 
                    new_pts[:,[4,5,6,7],2] = heights.unsqueeze(1).repeat(1,4).double().to(points.device)
            
        else:
            print("No heights were input")
            return
        
        if refine_heights:
            template_boxes = self.space_to_im(new_pts,name = name,times = times)
            heights_new = self.height_from_template(template_boxes, heights, points.view(d,8,3))
            new_pts[:,[4,5,6,7],2] = heights_new.unsqueeze(1).repeat(1,4).double().to(points.device)
            
        return new_pts
    
    @safe_name
    def _sp_im(self,points, name = None, times = None):
       """
       Projects 3D space points into image/correspondence using P:
           new_pts = P x points T  ---> [dm,3] T = [3,4] x [4,dm]
       performed by flattening batch dimension d and object point dimension m together
       
       name      - list of correspondence key names
       times     - None or float or [float] - time or times for positions     
       points    - [d,m,3] array of points in 3-space, m is probably 8
       RETURN:     [d,m,2] array of points in 2-space
       """
       
       d = points.shape[0]
       if d == 0:
           return torch.empty(0,8,2)
       
       if name is None:
           name = list(self.correspondence.keys())[0]
       elif type(name) == list and len(name[0].split("_")) == 1:
           name = self.get_direction(points,name)[0]
       n_pts = points.shape[1]
       # get directions and append to names
       
       
       
       
           
       # convert points into size [dm,4]
       points = points.view(-1,3)
       points = torch.cat((points.double(),torch.ones([points.shape[0],1],device = points.device).double()),1) # add 4th row
       
       
       # project into [dm,3]
       if type(name) == list:
               P = self.get_P(name,times).double()
               P = P.unsqueeze(1).repeat(1,n_pts,1,1).reshape(-1,3,4).to(points.device)
               points = points.unsqueeze(1).transpose(1,2)
               new_pts = torch.bmm(P,points).squeeze(2)
       else:
           points = torch.transpose(points,0,1).double()
           P = self.get_P(name,times).to(points.device).double()
           new_pts = torch.matmul(P,points).transpose(0,1)
       
       # divide each point 0th and 1st column by the 2nd column
       new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
       new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
       
       # drop scale factor column
       new_pts = new_pts[:,:2] 
       
       # reshape to [d,m,2]
       new_pts = new_pts.view(d,-1,2)
       return new_pts 
    
    
    
    def im_to_space(self,points, name = None,heights = None,classes = None,times = None,refine_heights = True):
        """
        Wrapper function on _im_sp necessary because it is not immediately evident 
        from points in image whether the EB or WB corespondence should be used
        
        points    - [d,m,2] array of points in image
        name      - list of correspondence key names
        heights   - [d] tensor of object heights, 0, or None (use classes)
        classes   - None or [d] tensor of object classes
        times     - None or float or [float] - time or times for positions   
        RETURN:     [d,m,3] array of points in space 
        """
        
        points_ds = points * self.downsample
        
        if heights is None:
            if classes is None: 
                raise IOError("Either heights or classes must not be None")
            else:
                heights = self.guess_heights(classes)
        if type(name) != list or len(name[0].split("_")) == 1:        
            boxes  = self._im_sp(points_ds,name = name, times = times, heights = 0)
            
            # get directions and append to names
            name = self.get_direction(boxes,name)[0]
            
        # recompute with correct directions
        boxes = self._im_sp(points_ds,name = name, heights = heights, times = times, refine_heights=refine_heights)
        
        
        return boxes
    
    def space_to_im(self, points, times = None, name = None):
        """
        Wrapper function on _sp_im necessary because it is not immediately evident 
        from points in image whether the EB or WB corespondence should be used
        
        name    - list of correspondence key names
        points  - [d,m,3] array of points in 3-space
        times     - None or float or [float] - time or times for positions   
        RETURN:   [d,m,2] array of points in 2-space
        """
        
        boxes  = self._sp_im(points,name = name,times = times)     
        boxes /= self.downsample
        
        return boxes
        
    def im_to_state(self,points, name = None, heights = None,times = None,refine_heights = True,classes = None):
        """
        Converts image boxes to roadway coordinate boxes
        points    - [d,m,2] array of points in image
        name      - list of correspondence key names
        heights   - [d] tensor of object heights
        times     - None or float or [float] - time or times for positions   
        RETURN:     [d,s] array of boxes in state space where s is state size (probably 6)
        """
        space_pts = self.im_to_space(points,name = name, classes = classes, times = times, heights = heights,refine_heights = refine_heights)
        return self.space_to_state(space_pts)
    
    def state_to_im(self,points,name = None,times = None):
        """
        Converts roadway coordinate boxes to image space boxes
        points    - [d,s] array of boxes in state space where s is state size (probably 6)
        name      - list of correspondence key names
        times     - None or float or [float] - time or times for positions   
        RETURN:   - [d,m,2] array of points in image
        """
        space_pts = self.state_to_space(points)
        return self.space_to_im(space_pts,name = name,times = times)
    
    def space_to_state(self,points):
        """        
        Conversion from state plane coordinates to roadway coordinates via the following steps:
            1. If spline points aren't yet cached, cache them
            2. Convert space points to L,W,H,x_back,y_center
            3. Search coarse grid for best fit point for each point
            4. Search fine grid for best offset relative to each coarse point
            5. Final state space obtained
        
        Note that by convention 3D box point ordering  = fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl and roadway coordinates reference back center of vehicle

        
        points - [d,m,3] 
        RETURN:  [d,s] array of boxes in state space where s is state size (probably 6)
        """
        points = points.clone() # deals with yellow line offset issue 

        
        # 2. Convert space points to L,W,H,x_back,y_center
        d = points.shape[0]
        n_pts = points.shape[1]
        
        if d == 0:
            return torch.empty([0,6], device = points.device)
        
        new_pts = torch.zeros([d,6],device = points.device)
        
        # rear center bottom of vehicle is (x,y)
        
        if n_pts == 8:
            # x is computed as average of two bottom rear points
            new_pts[:,0] = (points[:,2,0] + points[:,3,0]) / 2.0
            
            # y is computed as average of two bottom rear points 
            new_pts[:,1] = (points[:,2,1] + points[:,3,1]) / 2.0

            
            # l is computed as avg length between bottom front and bottom rear
            #new_pts[:,2] = torch.abs ( ((points[:,0,0] + points[:,1,0]) - (points[:,2,0] + points[:,3,0]))/2.0 )
            new_pts[:,2] = torch.pow((points[:,[0,1,4,5],:] - points[:,[2,3,6,7],:]).mean(dim = 1),2).sum(dim = 1).sqrt()
            
            # w is computed as avg length between botom left and bottom right
            new_pts[:,3] = torch.pow((points[:,[0,2,4,6],:] - points[:,[1,3,5,7],:]).mean(dim = 1),2).sum(dim = 1).sqrt()

            # h is computed as avg length between all top and all bottom points
            new_pts[:,4] = torch.mean(torch.abs( (points[:,0:4,2] - points[:,4:8,2])),dim = 1)
        
        else:
            new_pts[:,0] = points[:,0,0]
            new_pts[:,1] = points[:,0,1]
            
        
        # direction is +1 if vehicle is traveling along direction of increasing x, otherwise -1
        directions = self.get_direction(points,)[1]
        new_pts[:,5] = directions #torch.sign( ((points[:,0,0] + points[:,1,0]) - (points[:,2,0] + points[:,3,0]))/2.0 ) 
        

        min_u = self.closest_spline_point(new_pts[:,:2])
        min_u = torch.from_numpy(min_u).clamp(min = 0)
        spl_x,spl_y = interpolate.splev(min_u,self.median_tck)
        spl_x,spl_y = torch.from_numpy(spl_x).to(points.device),torch.from_numpy(spl_y).to(points.device)
        min_dist = torch.sqrt((spl_x - new_pts[:,0])**2 + (spl_y - new_pts[:,1])**2)
        
        new_pts[:,0] = min_u
        new_pts[:,1] = min_dist
        
        # if direction is -1 (WB), y coordinate is negative
        new_pts[:,1] *= new_pts[:,5]
        
        new_pts[:,5] *= self.polarity
        
        if self.yellow_offsets is not None:
            # shift so that yellow lines have constant y-position
            bins = (new_pts[:,0] / 10).int()
            bins = torch.clamp(bins,min = 0, max = len(self.yellow_offsets["WB"])-1)

            bins = bins.data.cpu().numpy()
            
            eb_offsets = self.yellow_offsets["EB"][bins] -12
            wb_offsets = self.yellow_offsets["WB"][bins] +12 
            eb_mask = torch.where(new_pts[:,1] > 0, 1, 0).to("cpu")
            wb_mask = 1- eb_mask
            
            yellow_offsets = wb_mask * wb_offsets + eb_mask * eb_offsets
            yellow_offsets = yellow_offsets.to(new_pts.device)
            new_pts[:,1] = new_pts[:,1] - yellow_offsets
            #new_pts[:,1] += 3
            
        # 5. Final state space obtained
        return new_pts
        
    def state_to_space(self,points):
        """
        Conversion from state plane coordinates to roadway coordinates via the following steps:
            1. get x-y coordinate of closest point along spline (i.e. v = 0)
            2. get derivative of spline at that point
            3. get perpendicular direction at that point
            4. Shift x-y point in that direction
            5. Offset base points in each constituent direction
            6. Add top points
            
        Note that by convention 3D box point ordering  = fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl and roadway coordinates reference back center of vehicle
        
        points - [d,s] array of boxes in state space where s is state size (probably 6)
        
        """
        points = points.clone() # deals with yellow line offset issue 
        
        
        
        
        #### WARNING, this is untested!!!
        points = points.view(-1,points.shape[-1])
        
        d = points.shape[0]
        
        try:
            points[:,5] *= self.polarity
        except:
            print("Error in state_to_space, points is of dimension: {}".format(points.shape))
            return torch.empty([0,8,3], device = points.device)
        
        if d == 0:
            return torch.empty([0,8,3], device = points.device)
        
        # 0. Un-offset points by yellow lines
        if self.yellow_offsets is not None:
            # shift so that yellow lines have constant y-position
            bins = (points[:,0] / 10).int()
            bins = torch.clamp(bins,min = 0, max = len(self.yellow_offsets["WB"])-1)

            bins = bins.data.cpu().numpy()

            eb_offsets = self.yellow_offsets["EB"][bins] -12
            wb_offsets = self.yellow_offsets["WB"][bins] +12 
            eb_mask = torch.where(points[:,1] > 0, 1, 0).to("cpu")
            wb_mask = 1- eb_mask
            
            yellow_offsets = wb_mask * wb_offsets + eb_mask * eb_offsets
            yellow_offsets = yellow_offsets.to(points.device)
            points[:,1] = points[:,1] + yellow_offsets 
            #points[:,1] -= 3
        
        

        # 1. get x-y coordinate of closest point along spline (i.e. v = 0)
        closest_median_point_x, closest_median_point_y = interpolate.splev(points[:,0].cpu(),self.median_tck)
        
        # 2. get derivative of spline at that point
        l_direction_x,l_direction_y          = interpolate.splev(points[:,0].cpu(),self.median_tck, der = 1)

        # 3. get perpendicular direction at that point
        #w_direction_x,w_direction_y          = -1/l_direction_x  , -1/l_direction_y
        w_direction_x,w_direction_y = l_direction_y,-l_direction_x
        
        # numpy to torch - this is not how you should write this but I was curious
        [closest_median_point_x,
        closest_median_point_y,
        l_direction_x,
        l_direction_y,
        w_direction_x,
        w_direction_y] = [torch.from_numpy(arr).to(points.device) for arr in [closest_median_point_x,
                                                           closest_median_point_y,
                                                           l_direction_x,
                                                           l_direction_y,
                                                           w_direction_x,
                                                           w_direction_y]]
        direction = torch.sign(points[:,1])
                                                            
        # 4. Shift x-y point in that direction
        hyp_l = torch.sqrt(l_direction_x**2 + l_direction_y **2)
        hyp_w = torch.sqrt(w_direction_x**2 + w_direction_y **2)
        
        # shift associated with the length of the vehicle, in x-y space
        x_shift_l    = l_direction_x /hyp_l * points[:,2] * direction
        y_shift_l    = l_direction_y /hyp_l * points[:,2] * direction
        
        # shift associated with the width of the vehicle, in x-y space
        x_shift_w    = w_direction_x/hyp_w * (points[:,3]/2.0) * direction
        y_shift_w    = w_direction_y/hyp_w * (points[:,3]/2.0) * direction
        
        # shift associated with the perp distance of the object from the median, in x-y space
        x_shift_perp = w_direction_x/hyp_w * points[:,1] # * direction # points already incorporates sign you dingus!!
        y_shift_perp = w_direction_y/hyp_w * points[:,1] # * direction
        
        # 5. Offset base points in each constituent direction
        
        new_pts = torch.zeros([d,4,3],device = points.device)
        
        # shift everything to median point
        new_pts[:,:,0] = closest_median_point_x.unsqueeze(1).expand(d,4) + x_shift_perp.unsqueeze(1).expand(d,4)
        new_pts[:,:,1] = closest_median_point_y.unsqueeze(1).expand(d,4) + y_shift_perp.unsqueeze(1).expand(d,4)
        
        # shift front points
        new_pts[:,[0,1],0] += x_shift_l.unsqueeze(1).expand(d,2)
        new_pts[:,[0,1],1] += y_shift_l.unsqueeze(1).expand(d,2)
        
        new_pts[:,[0,2],0] += x_shift_w.unsqueeze(1).expand(d,2)
        new_pts[:,[0,2],1] += y_shift_w.unsqueeze(1).expand(d,2)
        new_pts[:,[1,3],0] -= x_shift_w.unsqueeze(1).expand(d,2)
        new_pts[:,[1,3],1] -= y_shift_w.unsqueeze(1).expand(d,2)
    
        #6. Add top points
        top_pts = new_pts.clone()
        top_pts[:,:,2] += points[:,4].unsqueeze(1).expand(d,4)
        new_pts = torch.cat((new_pts,top_pts),dim = 1)
        
        
        
        #test
        #test_direction = self.get_direction(new_pts)[1]
        
        return new_pts
    
    def get_direction(self,points,name = None,method = "angle"):
        """
        Find closest point on spline. use relative x_coordinate difference (in state plane coords) to determine whether point is below spline (EB) or above spline (WB)
        
        
        points    - [d,m,3] array of points in space
        name      - list of correspondence key names. THey are not needed but if supplied the correct directions will be appended to them
        RETURN:   - [d] list of names with "EB" or "WB" added, best guess of which side of road object is on and which correspondence should be used
                  - [d] tensor of int with -1 if "WB" and 1 if "EB" per object
        """
        mean_points = torch.mean(points, dim = 1).cpu()
        min_u  = self.closest_spline_point(mean_points)
        
        if method == "x_position":
            spl_x,_ = interpolate.splev(min_u, self.median_tck)
            
            direction = (torch.sign(torch.from_numpy(spl_x) - mean_points[:,0]).int().to(points.device))
        
        elif method == "angle":
            spl_x,spl_y         =  interpolate.splev(min_u    , self.median_tck)
            x_forward,y_forward =  interpolate.splev(min_u+250, self.median_tck)
            
            # now we have 3 points defining 2 vectors and one directioned angle
            # vector A -  closest point -> 250 feet ahead on spline 
            vec_A = torch.from_numpy(x_forward - spl_x) , torch.from_numpy( y_forward - spl_y)
            
            # vector B - closest point -> point
            vec_B = mean_points[:,0] - spl_x, mean_points[:,1] - spl_y
            
            # if angle AB is in [0,180], object is on WB sidez, otherwise on EB side
            # from https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
            angle = torch.atan2( vec_A[0]*vec_B[1] - vec_A[1]*vec_B[0], vec_A[0]*vec_B[0] + vec_A[1]*vec_B[1] )
            direction =  -1* torch.sign(angle).int().to(points.device)
        
        d_list = ["dummy which should never appear","EB","WB"]
        string_direction = [d_list[di.item()] for di in direction]
        
    
        new_name = None                        
        if name is not None:
            
            if type(name) == str:
                new_name = name + "_{}".format(string_direction[0])
            
            elif type(name) == list:
                new_name = []
                for n_idx in range(len(name)):
                    if "C03" in name[n_idx]:
                        new_name.append(name[n_idx] + "_EB")
                    elif "C04" in name[n_idx]:
                        new_name.append(name[n_idx] + "_WB")
                    else: 
                        new_name.append(name[n_idx] + "_{}".format(string_direction[n_idx]))
            
        return new_name, direction
    
    
    def guess_heights(self,classes):
        """
        classes - [d] vector of string class names
        
        returns - [d] vector of float object height guesses
        """
        
        heights = torch.zeros(len(classes))
        
        for i in range(len(classes)):
            try:
                heights[i] = self.class_heights[classes[i]]
            except KeyError:
                heights[i] = self.class_heights["other"]
            
        return heights
    
    def height_from_template(self,template_boxes,template_space_heights,boxes):
        """
        Predicts space height of boxes in image space. Given a space height and 
        the corresponding image box (and thus image height), the relationship 
        between heights in different coordinate systems should be roughly estimable. 
        This strategy is used to guess the heights of the second set of boxes in
        image space according to : 
            template_im_heights:template_space_heights = new_im_heights:new_box heights
            
        template_boxes - [d,m,2,] array of points corresponding to d object boxes 
                         (typical usage would be to use boxes from previous frame
                         or apriori box predictions for current frame))
        template_space_heights - [d] array of corresponding object heights in space
        boxes - [d,m,2] array of points in image
        
        returns
        
        height - [d] array of object heights in space
        """
        
        # get rough heights of objects in image
        template_top = torch.mean(template_boxes[:,4:8,:],dim = 1)
        template_bottom = torch.mean(template_boxes[:,0:4,:],dim = 1)
        template_im_height = torch.sum(torch.sqrt(torch.pow((template_top - template_bottom),2)),dim = 1) *self.downsample
        template_ratio = template_im_height / template_space_heights.to(boxes.device)
        
        box_top    = torch.mean(boxes[:,4:8,:2],dim = 1)
        box_bottom = torch.mean(boxes[:,0:4,:2],dim = 1)
        box_height = torch.sum(torch.sqrt(torch.pow((box_top - box_bottom),2)),dim = 1)
        
        height = box_height / template_ratio
        return height

    
    def space_to_gps(self,points):
        """
        Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
        Transform is expected to be accurate within ~2 feet
        
        points array or tensor of size [n_pts,1,3] (X,Y,Z in state plane feet)
        returns out - array or tensor of size [n_pts,2] (Lat Long)
        """
        try:
            points = torch.clone(points[:,0,0:2])
            
            #wgs84=pyproj.CRS("EPSG:4326")
            #tnstate=pyproj.CRS("epsg:2274")
            transformer = pyproj.Transformer.from_crs("epsg:2274","EPSG:4326")
            out = transformer.transform(points[:,0].data.numpy(),points[:,1].data.numpy())
            out = np.array(out).transpose(1,0)
            out = torch.from_numpy(out)
            return out
        
        except NameError:
            print("Error: pyproj not imported or installed, returning None")
            return None
    
    def gps_to_space(self,points):
        """
        Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
        Transform is expected to be accurate within ~2 feet
        
        points array or tensor of size [n_pts,2]
        returns out - array or tensor of size [n_pts,1,3]
        """
        try:
            transformer = pyproj.Transformer.from_crs("EPSG:4326","epsg:2274")
            out = transformer.transform(points[:,0].data.numpy(),points[:,1].data.numpy())
            out = np.array(out).transpose(1,0)
            
            out = torch.from_numpy(out)
            out = torch.cat((out,torch.zeros(out.shape[0],1)),dim = 1)
            out = out.unsqueeze(1)
                
            return out
        
        except NameError:
            print("Error: pyproj not imported or installed, returning None")
            return None
    
    #%% Plotting Functions
    
    def plot_boxes(self,im,boxes,color = (255,255,255),labels = None,thickness = 1, ORIGIN = True):
        """
        As one might expect, plots 3D boxes on input image
        
        im - cv2 matrix-style image
        boxes - [d,8,2] array of image points where d indexes objects
        color - 3-tuple specifying box color to plot
        """
                
        DRAW = [[0,1,2,0,1,0,0,0], #bfr
                [0,0,0,1,0,1,0,0], #bfl
                [0,0,0,3,0,0,4,1], #bbr
                [0,0,0,0,0,0,1,1], #bbl
                [0,0,0,0,0,1,1,0], #tfr
                [0,0,0,0,0,0,0,1], #tfl
                [0,0,0,0,0,0,0,1], #tbr
                [0,0,0,0,0,0,0,0]] #tbl
        
        DRAW_BASE = [[0,1,1,1], #bfr
                     [0,0,1,1], #bfl
                     [0,0,0,1], #bbr
                     [0,0,0,0]] #bbl
        
        for idx, bbox_3d in enumerate(boxes):
            
            # check whether box mostly falls within frame
            if torch.min(bbox_3d[:,0]) < -1000 or torch.max(bbox_3d[:,0]) > 3840+1000 or torch.min(bbox_3d[:,1]) < -1000 or torch.max(bbox_3d[:,1]) > 2160+1000:
                continue
            
            if type(color) == np.ndarray:
                c = (int(color[idx,0]),int(color[idx,1]),int(color[idx,2]))
            else:
                c = color
            

            
            for a in range(len(bbox_3d)):
                ab = bbox_3d[a]
                for b in range(a,len(bbox_3d)):
                    bb = bbox_3d[b]
                    if DRAW[a][b] == 1:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),c,thickness)
                    elif DRAW[a][b] == 2:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(0,255,0),thickness)
                    elif DRAW[a][b] == 3:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(255,255,0),thickness)
                    elif DRAW[a][b] == 4:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(0,0,255),thickness)
        
            if labels is not None:
                label = labels[idx]
                left  = bbox_3d[0,0]
                top   = bbox_3d[0,1]
                im    = cv2.putText(im,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
                im    = cv2.putText(im,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)
            
            # draw origin
            if ORIGIN:
                bbl = bbox_3d[2,:]
                bbl = int(bbl[0]),int(bbl[1])
                im = cv2.circle(im,bbl,2,(255,255,255),-1)
            
        return im
        
    def plot_state_boxes(self,im,boxes,times = None,color = (255,255,255),labels = None, thickness = 1, name = None):
        """
        Wraps plot_boxes for state boxes by first converting from state (roadway coordinates) to image coordinates
        times     - None or float or [float] - time or times for positions   
        """
        im_boxes = self.state_to_im(boxes, name = name,times = times)
        return self.plot_boxes(im,im_boxes,color = color,labels = labels, thickness = thickness)
    
    def plot_points(self,im,points,times = None, color = (0,255,0)):
        """
        Lazily, duplicate each point 8 times as a box with 0 l,w,h then call plot_boxes
        points -  [d,2] array of x,y points in roadway coordinates / state 
        """
        rep_points = torch.cat(points,torch.zeros(points.shape[0],3),dim = 1)
        space_points = self.state_to_space(rep_points)
        self.plot_space_points(im,space_points,times = times,color = color)
    
    def plot_space_points(self,im,points,times = None,color = (255,0,0), name = None):
        """
        points -  [d,n,3] array of x,y points in roadway coordinates / state 
        times     - None or float or [float] - time or times for positions   
        """
        
        im_pts = self.space_to_im(points, times = times, name = name)
        
        for point in im_pts:
            cv2.circle(im,(int(point[0]),int(point[1])),2,color,-1)
        
        cv2.imshow("frame",im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
   
        
    def plot_homography(self,
                        im,
                        camera_name):
        False1 = True
        # for each correspondence (1 or 2)
        for suffix in ["_WB","_EB"]:
            corr = camera_name + suffix
            
            if False1:
                if "space_pts" in self.correspondence[corr].keys():
                    state_plane_pts = torch.from_numpy(self.correspondence[corr]["space_pts"])
                    state_plane_pts = torch.cat((state_plane_pts,torch.zeros([state_plane_pts.shape[0],1])),dim = 1)
                    road_pts = self.space_to_state(state_plane_pts.unsqueeze(1))
                    xmin = torch.min(road_pts[:,0])
                    xmax = torch.max(road_pts[:,0])
                
                    if suffix == "_EB":
                        ymin = 12
                        ymax = 80
                    else:
                        ymin = -60
                        ymax = -10
                    
                else:
                    pass # will write this when I get the new data format for v3 system    
                
                #generate grid -this is sloppy and could be tensorized, for which I apologize
                xmin = 0
                xmax = 50000
                if suffix == "_EB":
                    ymin = 12
                    ymax = 80
                else:
                    ymin = -60
                    ymax = -10
                    
                pts = []
                for x in np.arange(xmin,xmax,40):
                    for y in np.arange(ymin,ymax,12):
                        pt = torch.tensor([x,y,0,0,10,torch.sign(torch.tensor(y))])
                        pts.append(pt)
                
                road_grid = torch.stack(pts)
                
                
                
                im_grid    = self.state_to_im(road_grid,name = corr)
            
            if False:
                # replace points with marker points
                
                
                data = pd.read_csv("/home/worklab/Downloads/test_export_markers.csv")
                points = torch.from_numpy(data[["State X","State Y"]].to_numpy())
                points = torch.cat((points,torch.zeros([points.shape[0],1])),dim = 1).unsqueeze(1)
                points = points.contiguous().expand(points.shape[0],8,3).contiguous()
                
                im_grid = self.space_to_im(points,name = corr)
            
            for i in range(len(im_grid)):
                p1 = int(im_grid[i,0,0]),int(im_grid[i,0,1])
                p2 = int(im_grid[i,4,0]),int(im_grid[i,4,1])
                color = (255,0,0) if suffix == "_EB" else (0,0,255)
                cv2.line(im,p1,p2,color,1)
                cv2.circle(im,p1,5,color,-1)

            # if False1:
            #     for t in range(0,len(self.correspondence[corr]["P_dynamic"])):
            #         ts = [float(t*self.hg_sec + self.hg_start_time) for _ in range(road_grid.shape[0])]
                    
            #         im_grid = self.state_to_im(road_grid,name= corr, times = ts)
                    
            #         for i in range(len(im_grid)):
            #             p1 = int(im_grid[i,0,0]),int(im_grid[i,0,1])
            #             p2 = int(im_grid[i,4,0]),int(im_grid[i,4,1])
            #             color = (100,0,0) if suffix == "_EB" else (0,0,255)
            #             cv2.line(im,p1,p2,color,1)

        
            cv2.imshow("frame",im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    def gen_geom(self,space_dir,rcs_version_number = "XX"):
        """
        Generates the geometry .json file for an RCS coordinate system. 
        
        output:
            dict with
            landmarks - 
            detectors - 
            gantries - 
            milemarkers - 
            poles - 
            rcs_extents - 
            rcs_extents_st - 
            offset - the rcs x-coordinate of the "magic" reference point (originally this point was the location of MM60 but this may not always be the case if the milemarker is moved)
                        this offset is used to rectify outside MM-based systems with the internal RCS
        """
                
       
        # load landmark data
        file = os.path.join(space_dir,"landmarks.csv")

        dataframe = pd.read_csv(os.path.join(space_dir,file))
        st_x = dataframe["x"].tolist()
        st_y = dataframe["y"].tolist()
        st_type = dataframe["type"].tolist()
        st_location = dataframe["location"].tolist()

        # convert all points into roadway coords
        space_data = torch.tensor([st_x,st_y,torch.zeros(len(st_x))]).permute(1,0).unsqueeze(1)
        road_data = self.space_to_state(space_data)[:,:2]
        names = [st_type[i] + "_" + st_location[i] for i in range(len(st_type))]
        
        underpasses = {}
        overpasses = {}
       
        for n_idx in range(len(names)):
           name = names[n_idx]
           
           if "under" in name:
               try:
                   underpasses[name.split("_")[2]].append(road_data[n_idx,0].item())
               except:
                   underpasses[name.split("_")[2]] = [road_data[n_idx,0].item()]
       
           if "over" in name:
               try:
                   overpasses[name.split("_")[2]].append(road_data[n_idx,0].item())
               except:
                   overpasses[name.split("_")[2]] = [road_data[n_idx,0].item()]   
        
        
       
        # load pole data
        file = os.path.join(space_dir,"poles.csv")

        dataframe = pd.read_csv(os.path.join(space_dir,file))
        st_x = dataframe["x"].tolist()
        st_y = dataframe["y"].tolist()
        pole = dataframe["pole-number"].tolist()
        
        space_data_pole = torch.tensor([st_x,st_y,torch.zeros(len(st_x))]).permute(1,0).unsqueeze(1)
        road_data_pole = self.space_to_state(space_data_pole)[:,:2]
        
        poles = {}
        
        for p_idx in range(len(pole)):
            p_name = pole[p_idx]
            poles[p_name] = [road_data_pole[p_idx,0].item(), road_data_pole[p_idx,1].item()]
        
        
        
        
        
        
        # load milemarker data
        file = os.path.join(space_dir,"milemarker.csv")

        # load all points
        dataframe = pd.read_csv(file)
        dataframe = dataframe[dataframe['point_id'].notnull()]
        
        mm = dataframe["milemarker"].tolist()
        st_x = dataframe["x"].tolist()
        st_y = dataframe["y"].tolist()
        
        # convert each state plane  point into roadway
        mm_space = torch.tensor([st_x,st_y,[0 for _ in range(len(st_x))]]).transpose(1,0)
        mm_space = mm_space.unsqueeze(1).expand(mm_space.shape[0],8,3)
        mm_state = self.space_to_state(mm_space)[:,:2]
        
        milemarkers = {}
        for midx,m in enumerate(mm):
            milemarkers[m] = [mm_state[midx,0].item(),mm_state[midx,1].item()]        
        
        
        
        
        # load gantry data
        file = os.path.join(space_dir,"gantries.csv")

        # load all points
        dataframe = pd.read_csv(file)
        dataframe = dataframe[dataframe['x'].notnull()]
        
        mm = dataframe["tdot_milemarker"].tolist()
        direction = dataframe["road_side"].tolist()
        st_x = dataframe["x"].tolist()
        st_y = dataframe["y"].tolist()
        
        # convert each state plane  point into roadway
        mm_space = torch.tensor([st_x,st_y,[0 for _ in range(len(st_x))]]).transpose(1,0)
        mm_space = mm_space.unsqueeze(1).expand(mm_space.shape[0],8,3)
        mm_state = self.space_to_state(mm_space)[:,:2]
        
        # get names
        gantry_names = [str(mm[m])+ "_" + str(direction[m]) for m in range(len(mm))]
        gantries_w = {}
        gantries_e = {}
        
        for gidx,g in enumerate(gantry_names):
            if "e" in g:
                gantries_e[g] = [mm_state[gidx,0].item(),mm_state[gidx,1].item()] 
            else:
                gantries_w[g] = [mm_state[gidx,0].item(),mm_state[gidx,1].item()] 
        
        
        
        
        # load detector data
        file = os.path.join(space_dir,"detectors.csv")

        # load all points
        dataframe = pd.read_csv(file)
        dataframe = dataframe[dataframe['x'].notnull()]
        
        mm = dataframe["tdot_milemarker"].tolist()
        direction = dataframe["road_side"].tolist()
        st_x = dataframe["x"].tolist()
        st_y = dataframe["y"].tolist()
        
        # convert each state plane  point into roadway
        mm_space = torch.tensor([st_x,st_y,[0 for _ in range(len(st_x))]]).transpose(1,0)
        mm_space = mm_space.unsqueeze(1).expand(mm_space.shape[0],8,3)
        mm_state = self.space_to_state(mm_space)[:,:2]
        
        # get names
        detector_names = [str(mm[m])+ "_" + str(direction[m]) for m in range(len(mm))]
        detectors = {}
        for didx,d in enumerate(gantry_names):
            detectors[d] = [mm_state[didx,0].item(),mm_state[didx,1].item()] 
        
        # generate rcs extents
        rcs_extents = self.median_u[[0,-1]]
        
        extst  = torch.from_numpy(rcs_extents).unsqueeze(1)
        extst  = torch.cat((extst,torch.zeros(extst.shape[0],5)),dim = 1)
        st_extents = self.state_to_space(extst)[:,0,:2].tolist()
        rcs_extents = rcs_extents.tolist()
        
        # generate magic number
        to_tdot_mm = milemarkers[60.0][0]
        in_feet = 60.0*5280
        magic_offset = [to_tdot_mm,in_feet]
        
        # save all
        geometry = { "overpass":overpasses,
                     "underpass":underpasses,
                     "poles":poles,
                     "milemarkers": milemarkers,
                     "gantries_e":gantries_e,
                     "gantries_w":gantries_w,
                     "detectors":detectors,
                     "rcs_extents":rcs_extents,
                     "state_extents":st_extents,
                     "mm60_offset":magic_offset
                     }
        
        output_path = os.path.join(space_dir,"rcs_{}_geom.json".format(rcs_version_number))
        with open(output_path,"w") as f:    
                  json.dump(geometry,f, sort_keys = True)
                  
        
        for name in poles:
            polex = poles[name][0]
            
            polemm = 60  + (polex - to_tdot_mm )/5280
            
            print(name, np.round(polemm,1))
        
    def gen_extents(self):
         cam_extents = {}
         for corr in self.correspondence.keys():
                     # get extents
                     pts = self.correspondence[corr]["FOV"]
                     pts = torch.from_numpy(np.array(pts))
                     pts = pts.unsqueeze(1).expand(pts.shape[0],8,2)
                     pts_road = self.im_to_state(pts,name = [corr for _ in pts],heights = torch.zeros(pts.shape[0]))
                     
                     
                     
                     minx = torch.min(pts_road[:,0]).item()
                     maxx = torch.max(pts_road[:,0]).item()
                     miny = torch.min(pts_road[:,1]).item()
                     maxy = torch.max(pts_road[:,1]).item()
                     
                     
                 
                     cam_extents[corr] = [minx,maxx,miny,maxy]
         
         
         with open("extents.json","w") as f:    
             json.dump(cam_extents,f, sort_keys = True)
         return cam_extents

    def gen_report(self,camera_list, outfile = None):
        
        headers = ["cam_side","dynamic","static","reference"]
        lines = []
        lines.append(headers)
        print(headers)
        for cam in camera_list:
            for side in ["_EB", "_WB"]:
                camside = cam+side
                if camside in self.correspondence.keys():
                    d_pass = 0
                    d_total = 0
                    for td in self.correspondence[camside]["P_dynamic"]:
                        if not np.isnan(td.sum()):
                            d_pass += 1
                        d_total += 1
                    if not np.isnan(self.correspondence[camside]["P_static"].sum()):
                        s_pass = True
                    else: s_pass = False
                    
                    if not np.isnan(self.correspondence[camside]["P_reference"].sum()):
                        r_pass = True
                    else: r_pass = False
                    
                    line = [camside,d_pass/d_total,s_pass,r_pass]
                
                else:
                    line = [camside,0,False,False]
                lines.append(line)
                print(line)
                
        if outfile is not None:    
            with open(outfile, 'w', newline="\n") as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerows(lines)
     
    # def _convert_landmarks(self,space_dir):
        
    #      output_path = "landmarks.json"
        
    #      file = os.path.join(space_dir,"landmarks.csv")
        
    #      # load relevant data
    #      dataframe = pd.read_csv(os.path.join(space_dir,file))
    #      st_x = dataframe["X"].tolist()
    #      st_y = dataframe["Y"].tolist()
    #      st_type = dataframe["type"].tolist()
    #      st_location = dataframe["location"].tolist()

    #      # convert all points into roadway coords

    #      space_data = torch.tensor([st_x,st_y,torch.zeros(len(st_x))]).permute(1,0).unsqueeze(1)
    #      road_data = self.space_to_state(space_data)[:,:2]
    #      names = [st_type[i] + "_" + st_location[i] for i in range(len(st_type))]
         
    #      file = os.path.join(space_dir,"poles.csv")
        
    #      # load relevant data
    #      dataframe = pd.read_csv(os.path.join(space_dir,file))
    #      st_x = dataframe["X"].tolist()
    #      st_y = dataframe["Y"].tolist()
    #      pole = dataframe["pole-number"].tolist()
         
    #      space_data_pole = torch.tensor([st_x,st_y,torch.zeros(len(st_x))]).permute(1,0).unsqueeze(1)
    #      road_data_pole = self.space_to_state(space_data_pole)[:,:2]
         
         
         
         
    #      underpasses = {}
    #      overpasses = {}
    #      poles = {}
         
    #      for p_idx in range(len(pole)):
    #          p_name = pole[p_idx]
    #          poles[p_name] = [road_data_pole[p_idx,0].item(), road_data_pole[p_idx,1].item()]
         
    #      for n_idx in range(len(names)):
    #          name = names[n_idx]
             
    #          if "under" in name:
    #              try:
    #                  underpasses[name.split("_")[2]].append(road_data[n_idx,0].item())
    #              except:
    #                  underpasses[name.split("_")[2]] = [road_data[n_idx,0].item()]
         
    #          if "over" in name:
    #              try:
    #                  overpasses[name.split("_")[2]].append(road_data[n_idx,0].item())
    #              except:
    #                  overpasses[name.split("_")[2]] = [road_data[n_idx,0].item()]   
         
    #      pass
    #      # store as JSON of points
         
    #      landmarks = {"overpass":overpasses,
    #                   "underpass":underpasses,
    #                   "poles":poles
    #                   }
    #      print(landmarks)
    #      with open(output_path,"w") as f:    
    #          json.dump(landmarks,f, sort_keys = True)
        
#%% MAIN        
    
if __name__ == "__main__":
    if False:
        aerial_ref_dir = "/home/worklab/Documents/i24/i24_imref/aerial/all_poles_aerial_labels"
        im_ref_dir     = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/wacv_hg_v1"
        save_path      = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/WACV2024_hg_save.cpkl"
        
        hg = I24_RCS(save_path = save_path,aerial_ref_dir = aerial_ref_dir, im_ref_dir = im_ref_dir,downsample = 1,default = "dynamic")
        hg.save(save_path)
        hg.hg_start_time = 0
        hg.hg_sec        = 10
        #hg.yellow_offsets = None
        
        im_dir = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/4k"
        for imf in os.listdir(im_dir):
            # if "P35C06" not in imf:
            #     continue
            
            im_path = os.path.join(im_dir,imf)
            im = cv2.imread(im_path)
            

            cam = imf.split(".")[0]
            try:
                hg.plot_homography(im,cam)
            except:
                print("Error on {}".format(imf))
                
    if False:
        rcs_base = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/WACV2024_hg_save.cpkl"
        hg_dir = "/home/worklab/Documents/temp_hg_files_for_dev/first_day_hg"
        test_hg = "/home/worklab/Documents/temp_hg_files_for_dev/hg_batch6_test.cpkl"
        
        
        
        test_hg = "/home/worklab/Downloads/hg_videonode1.cpkl"
        hg = I24_RCS(save_path = test_hg)
        names = list(hg.correspondence.keys())
        for name in names:
            hg.correspondence.pop(name,None)
        #hg.save("rcs_base.cpkl")
        hg.load_correspondences(hg_dir)
        #hg.save("hg_batch6_test.cpkl")
        
        
        im_dir = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/4k"
        ims = os.listdir(im_dir)
        ims.sort()
        for imf in ims:
            # if "P35C06" not in imf:
            #     continue
            
            im_path = os.path.join(im_dir,imf)
            im = cv2.imread(im_path)
            
        
            cam = imf.split(".")[0]
            try:
                hg.plot_homography(im,cam)
            except:
                print("Error on {}".format(imf))
        
    if True:
        aerial_ref_dir = "/home/worklab/Documents/coordinates_3.0/aerial_ref_3.0"
        im_ref_dir = None #"/home/worklab/Documents/coordinates_3.0/cam_ref_3.0"
        #save_path = "/home/worklab/Documents/coordinates_3.0/hg_664538e4b476f991aef3d7eb.cpkl"
        save_path = "/home/worklab/Documents/i24/i24_rcs/test.cpkl"
        #save_path = None
        hg = I24_RCS(save_path = save_path,aerial_ref_dir = aerial_ref_dir, im_ref_dir = im_ref_dir,downsample = 1,default = "reference")
        hg.gen_geom(aerial_ref_dir,rcs_version_number="3-0")
        