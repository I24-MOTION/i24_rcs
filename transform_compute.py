import cv2
# import time
import os
import numpy as np
# import csv
# import string
# import re
# import copy
import _pickle as pickle
import torch
# import PyNvCodec as nvc
# import PytorchNvCodec as pnvc
# import torchvision.transforms.functional as F
# from scipy.spatial import ConvexHull
# import itertools
# import sys

import pandas as pd
from scipy import interpolate



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



def get_all_dash_points(direction):
    ae_x = []
    ae_y = []
    ae_id = []
    
    aerial_im_dir = "/home/derek/Documents/i24/i24_homography/aerial/all_poles_aerial_labels"

    for file in os.listdir(aerial_im_dir):
        if direction.lower() not in file:
            continue
        
        # load all points
        dataframe = pd.read_csv(os.path.join(aerial_im_dir,file))
        try:
            dataframe = dataframe[dataframe['point_pos'].notnull()]
            attribute_name = file.split(".csv")[0].split("_")[1]
            
            if attribute_name not in ["d1","d2","d3","d4","d5","hov","msg","gzc"]:
                continue
            
            feature_idx = dataframe["point_id"].tolist()
            st_id = [attribute_name + "_" + item for item in feature_idx]
            
            st_x = dataframe["st_x"].tolist()
            st_y = dataframe["st_y"].tolist()
        
            ae_x  += st_x
            ae_y  += st_y
            ae_id += st_id
        except:
            dataframe = dataframe[dataframe['side'].notnull()]
            attribute_name = file.split(".csv")[0].split("_")[1]
            
            if attribute_name not in ["d1","d2","d3","d4","d5","hov","msg","gzc"]:
                continue
            
            feature_idx = dataframe["id"].tolist()
            side        = dataframe["side"].tolist()
            st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
            
            st_x = dataframe["st_x"].tolist()
            st_y = dataframe["st_y"].tolist()
        
            ae_x  += st_x
            ae_y  += st_y
            ae_id += st_id

    return ae_x, ae_y, ae_id

def compute_correspondences(cameras, direction = "WB", ADD_PROJ = True):
    all_correspondences = {}
    
    # specify paths to aerial imagery point files
    aerial_im_dir = "/home/derek/Documents/i24/i24_homography/aerial/all_poles_aerial_labels"
    
    max_proj_error = 1
    scale_factor = 0.5
    
    #%% ### State space, do once
    
    ae_x = []
    ae_y = []
    ae_id = []
    
    
    for file in os.listdir(aerial_im_dir):
        if direction.lower() not in file:
            continue
        
        # load all points
        dataframe = pd.read_csv(os.path.join(aerial_im_dir,file))
        try:
            dataframe = dataframe[dataframe['point_pos'].notnull()]
            attribute_name = file.split(".csv")[0]
            feature_idx = dataframe["point_id"].tolist()
            st_id = [attribute_name + "_" + item for item in feature_idx]
            
            st_x = dataframe["st_x"].tolist()
            st_y = dataframe["st_y"].tolist()
        
            ae_x  += st_x
            ae_y  += st_y
            ae_id += st_id
        except:
            dataframe = dataframe[dataframe['side'].notnull()]
            attribute_name = file.split(".csv")[0]
            feature_idx = dataframe["id"].tolist()
            side        = dataframe["side"].tolist()
            st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
            
            st_x = dataframe["st_x"].tolist()
            st_y = dataframe["st_y"].tolist()
        
            ae_x  += st_x
            ae_y  += st_y
            ae_id += st_id
    
    
    # Find a-d end point of all d2 lane markers
    d2 = {}
    d3 = {}
    
    ae_spl_x = []
    ae_spl_y = []
    
    for i in range(len(ae_x)):
        if "d2" in ae_id[i]:
            if ae_id[i].split("_")[-1] in ["a","d"]:
                num = ae_id[i].split("_")[-2]
                if num not in d2.keys():
                    d2[num] = [(ae_x[i],ae_y[i])]
                else:
                    d2[num].append((ae_x[i],ae_y[i]))
        elif "d3" in ae_id[i]:
            if ae_id[i].split("_")[-1] in ["a","d"]:
                num = ae_id[i].split("_")[-2]
                if num not in d3.keys():
                    d3[num] = [(ae_x[i],ae_y[i])]
                else:
                    d3[num].append((ae_x[i],ae_y[i]))
                    
        elif "yeli" in ae_id[i]:
            ae_spl_x.append(ae_x[i])
            ae_spl_y.append(ae_y[i])
        
            
    
    # stack d2 and d3 into arrays            
    d2_ids = []
    d2_values = []
    for key in d2.keys():
        val = d2[key]
        d2_ids.append(key)
        d2_values.append(   [(val[0][0] + val[1][0])/2.0   ,   (val[0][1] + val[1][1])/2.0     ])
    
    d3_ids = []
    d3_values = []
    for key in d3.keys():
        val = d3[key]
        d3_ids.append(key)
        d3_values.append(   [(val[0][0] + val[1][0])/2.0   ,   (val[0][1] + val[1][1])/2.0     ])
    
    d2_values = torch.from_numpy(np.stack([np.array(item) for item in d2_values]))
    d3_values = torch.from_numpy(np.stack([np.array(item) for item in d3_values]))
    
    d2_exp = d2_values.unsqueeze(1).expand(d2_values.shape[0],d3_values.shape[0],2)
    d3_exp = d3_values.unsqueeze(0).expand(d2_values.shape[0],d3_values.shape[0],2)
    
    dist = torch.sqrt(torch.pow(d2_exp - d3_exp , 2).sum(dim = -1))
    
    min_matches = torch.min(dist, dim = 1)[1]
    
    if ADD_PROJ:
        
        try:
            with open("ae_cache_{}.cpkl".format(direction),"rb") as f:
                additional_points = pickle.load(f)
        except:
            # For each d2 lane marker, find the closest d3 lane marker
            proj_lines = []
            
            for i in range(len(min_matches)):
                j = min_matches[i]
                pline = [d3_values[j],d2_values[i],d3_ids[j],d2_ids[i]]
                proj_lines.append(pline)
            
            
            
            # compute the yellow line spline in state plane coordinates
            
            ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
            ae_data = ae_data[:,np.argsort(ae_data[1,:])]
            
            ae_tck, ae_u = interpolate.splprep(ae_data, s=0, per=False)
            
            span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
            ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*scale_factor)), ae_tck)
        
            additional_points = []
            # for each d2 lane marker, find the intersection between the d2-d3 line and the yellow line spline
            for p_idx, proj_line in enumerate(proj_lines):
                print("On proj line {} of {}".format(p_idx,len(proj_lines)))
                min_dist = np.inf
                min_point = None
                line = [proj_line[0][0],proj_line[0][1],proj_line[1][0],proj_line[1][1]]
                for i in range(len(ae_x_prime)):
                    point = [ae_x_prime[i],ae_y_prime[i]]
                    
                    dist = line_to_point(line, point)
                    if dist < min_dist:
                        min_dist = dist
                        min_point = point
                if min_dist > 1:
                    print("Issue")
                else:
                    name = "{}_{}".format(proj_line[2],proj_line[3])
                    min_point.append(name)
                    additional_points.append(min_point)
                    
            with open("ae_cache_{}.cpkl".format(direction),"wb") as f:
                pickle.dump(additional_points,f)
                
        
        for point in additional_points:
            ae_x.append(point[0])
            ae_y.append(point[1])
            ae_id.append(point[2])
            
    
    #%% ### Image space, do once per camera
    for camera in cameras:
        # specify path to camera imagery file
        cam_data_path = "/home/derek/Documents/i24/i24_homography/data_real/{}.cpkl".format(camera)
        cam_im_path   = "/home/derek/Documents/i24/i24_homography/data_real/{}.png".format(camera)
        
        # load all points
        with open(cam_data_path, "rb") as f:
            im_data = pickle.load(f)
            
        
        # get all non-curve matching points
        point_data = im_data[direction]["points"]
        filtered = filter(lambda x: x[2].split("_")[1] not in ["yeli","yelo","whli","whlo"],point_data)
        im_x  = []
        im_y  = []
        im_id = []
        for item in filtered:
            im_x.append(item[0])
            im_y.append(item[1])
            im_id.append(item[2])
        
        if len(im_x) == 0:
            continue
        
        if ADD_PROJ:
        
            # compute the yellow line spline in image coordinates
            curve_data = im_data[direction]["curves"]
            filtered = filter(lambda x: "yeli" in x[2], curve_data)
            x = []
            y = []
            for item in filtered:
                x.append(item[0])
                y.append(item[1])
            data = np.stack([np.array(x),np.array(y)])
            data = data[:,np.argsort(data[0,:])]
            
            tck, u = interpolate.splprep(data, s=0, per=False)
            x_prime, y_prime = interpolate.splev(np.linspace(0, 1, 4000), tck)
            
            if False:
                im = cv2.imread(cam_im_path)
                for i in range(len(x_prime)):
                    cv2.circle(im,(int(x_prime[i]),int(y_prime[i])), 2, (255,0,0,),-1)
                    
                cv2.imshow("frame",im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            # find all d2 and d3 points
        
            # find the intersection of each d2d3 line and the yellow line spline for d2-d3 pairs in image
            d2 = {}
            d3 = {}
            for i in range(len(im_x)):
                if "d2" in im_id[i]:
                    if im_id[i].split("_")[-1] in ["a","d"]:
                        num = im_id[i].split("_")[-2]
                        if num not in d2.keys():
                            d2[num] = [(im_x[i],im_y[i])]
                        else:
                            d2[num].append((im_x[i],im_y[i]))
                elif "d3" in im_id[i]:
                    if im_id[i].split("_")[-1] in ["a","d"]:
                        num = im_id[i].split("_")[-2]
                        if num not in d3.keys():
                            d3[num] = [(im_x[i],im_y[i])]
                        else:
                            d3[num].append((im_x[i],im_y[i]))
                
        
            # stack d2 and d3 into arrays            
            d2_ids = []
            d2_values = []
            for key in d2.keys():
                d2[key] = [(d2[key][0][0] + d2[key][1][0])/2.0  , (d2[key][0][1] + d2[key][1][1])/2.0 ] 
            
            d3_ids = []
            d3_values = []
            for key in d3.keys():
                d3[key] = [(d3[key][0][0] + d3[key][1][0])/2.0  , (d3[key][0][1] + d3[key][1][1])/2.0 ] 
            
            additional_im_points = []
            for proj_point in additional_points:
                
                
                d3_id = proj_point[2].split("_")[0]
                d2_id = proj_point[2].split("_")[1]
                
                if d3_id not in d3.keys() or d2_id not in d2.keys():
                    continue
                
                im_line = [d3[d3_id][0], d3[d3_id][1], d2[d2_id][0], d2[d2_id][1]]
                
                min_dist = np.inf
                min_point = None
                for i in range(len(x_prime)):
                    point = [x_prime[i],y_prime[i]]
        
                    dist = line_to_point(im_line, point)
                    if dist < min_dist:
                        min_dist = dist
                        min_point = point
                if min_dist > 2:
                    print("Issue")
                else:
                    name = proj_point[2]
                    min_point.append(name)
                    additional_im_points.append(min_point)
                    
            for point in additional_im_points:
                im_x.append(point[0])
                im_y.append(point[1])
                im_id.append(point[2])
                
                
                
        
            
        
        #%% ### Joint
        
        # assemble ordered list of all points visible in both image and space
        
        include_im_x  = []
        include_im_y  = []
        include_im_id = []
        
        include_ae_x  = []
        include_ae_y  = []
        include_ae_id = []
        
        for i in range(len(ae_id)):
            for j in range(len(im_id)):
                
                if ae_id[i] == im_id[j]:
                    include_im_x.append(  im_x[j])
                    include_im_y.append(  im_y[j])
                    include_im_id.append(im_id[j])
                    
                    include_ae_x.append(  ae_x[i])
                    include_ae_y.append(  ae_y[i])
                    include_ae_id.append(ae_id[i])
        
        
        
        # compute homography
        vp = im_data[direction]["z_vp"]
        corr_pts = np.stack([np.array(include_im_x),np.array(include_im_y)]).transpose(1,0)
        space_pts = np.stack([np.array(include_ae_x),np.array(include_ae_y)]).transpose(1,0)
        
        
        cor = {}
        cor["vps"] = vp
        cor["corr_pts"] = corr_pts
        cor["space_pts"] = space_pts
        
        cor["H"],_     = cv2.findHomography(corr_pts,space_pts)
        cor["H_inv"],_ = cv2.findHomography(space_pts,corr_pts)
        
        
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

        
        #%% find best Z -scaling

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
            
        
        print("Best Error: {}".format(best_error))
  


        
        
        
        
        cor["state_plane_pts"] = [include_ae_x,include_ae_y,include_ae_id]
        cor_name = "{}_{}".format(camera,direction)
        all_correspondences[cor_name] = cor

    return all_correspondences


if __name__ == "__main__":

    test = compute_correspondences(["P08C01","P08C02","P08C03","P08C04"],direction = "WB")


