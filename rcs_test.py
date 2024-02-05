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
import json
import matplotlib.pyplot as plt

from scipy import interpolate

from rcs import I24_RCS

 #%% Testing Functions
 
def test_transformation(hg,im_dir):
     """
     A series of tests will be run
     
     1. Transform a set of image points into space then state, then back to space then image
        - get average and max error (in pixels)
        
     2. Transform a set of randomly created boxes into space then image, then back to space then state
        - get average state error (in feet)
        
     3. Transform the same set of boxes back into image, and get average pixel error
     """
     
     if True:
         print("___________________________________")
         print("Test 0: Get Direction Test")
         print("___________________________________")
         
         correct = 0
         EB_incorrect = 0 # points on EB side that are incorrectly called WB
         WB_incorrect = 0 # points on WB side that are incorrectly called EB
         
         for name in hg.correspondence.keys():
             corr = hg.correspondence[name]
             direction = name.split("_")[1]
             #print(direction)
             space_pts = torch.from_numpy(corr["space_pts"]).unsqueeze(1)
             space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1,1])), dim = -1)
             
             pred = hg.get_direction(space_pts)[1]
             
             EB = torch.where(pred == 1,1,0).sum()
             WB = torch.where(pred == -1,1,0).sum()
             
             if direction == "EB":
                 correct += EB
                 if "c3" not in name and "c4" not in name:
                     EB_incorrect += WB

             elif direction == "WB":
                 correct += WB
                 if "c3" not in name and "c4" not in name:
                     WB_incorrect += EB
                 
         print("Correct side: {},   EB incorrect: {}, WB incorrect: {}".format(correct,EB_incorrect,WB_incorrect))
     
     ### Project each aerial imagery point into pixel space and get pixel error
     if True:
         print("___________________________________")
         print("Test 1: Pixel Reprojection Error")
         print("___________________________________")
         start = time.time()
         running_error = []
         for name in hg.correspondence.keys():
             try:
                 corr = hg.correspondence[name]
                 #name = name.split("_")[0]
                 
                 space_pts = torch.from_numpy(corr["space_pts"]).unsqueeze(1)
                 space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1,1])), dim = -1)
                 
                 im_pts    = torch.from_numpy(corr["corr_pts"])
                 
                 #name = name.split("_")[0]
                 namel = [name for _ in range(len(space_pts))]
                 
                 proj_space_pts = hg.space_to_im(space_pts,name = namel).squeeze(1)
                 error = torch.sqrt(((proj_space_pts - im_pts)**2).sum(dim = 1)).mean()
                 
                 if torch.isnan(error):
                     pass #print("Ill-defined homographies for camera {}".format(name))
                 else:
                     running_error.append(error)   
             except:
                pass
         end = time.time() - start
         print("Average Pixel Reprojection Error across all homographies: {}px in {} sec\n".format(sum(running_error)/len(running_error),end))
             
     
     ### Project each camera point into state plane coordinates and get ft error
     if True:
         print("___________________________________")
         print("Test 2: State Reprojection Error")
         print("___________________________________")
         running_error = []
         
         all_im_pts = []
         all_cam_names = []
         for name in hg.correspondence.keys():
             try:
                 corr = hg.correspondence[name]
                 #name = name.split("_")[0]
     
                 
                 space_pts = torch.from_numpy(corr["space_pts"])
                 space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1])), dim = -1)
     
                 im_pts    = torch.from_numpy(corr["corr_pts"]).unsqueeze(1).float()
                 namel = [name for _ in range(len(space_pts))]
     
                 all_im_pts.append(im_pts)
                 all_cam_names += namel
     
                 proj_im_pts = hg.im_to_space(im_pts,name = namel, heights = 0, refine_heights = False).squeeze(1)
                 
                 error = torch.sqrt(((proj_im_pts - space_pts)**2).sum(dim = 1)).mean()
                 #print("Mean error for {}: {}ft".format(name,error))
                 if torch.isnan(error):
                     pass # print("Ill-defined homographies for camera {}".format(name))
                 else:
                     running_error.append(error)   
             except:
                pass
         print("Average Space Reprojection Error across all homographies: {}ft\n".format(sum(running_error)/len(running_error)))
         

     
     if True:
         ### Create a random set of boxes
         boxes = torch.rand(1000,6) 
         boxes[:,0] = boxes[:,0] * (hg.median_u[-1] - hg.median_u[0]) + hg.median_u[0]
         boxes[:,1] = boxes[:,1] * 120 - 60
         boxes[:,2] *= 60
         boxes[:,3] *= 10
         boxes[:,4] *= 10
         boxes[:,5] = hg.polarity * torch.sign(boxes[:,1])

         
         space_boxes = hg.state_to_space(boxes)
         
         #gps test
         gps = hg.space_to_gps(space_boxes)
         space_back = hg.gps_to_space(gps)
         
         repro_state_boxes = hg.space_to_state(space_boxes)
         
         error = torch.abs(repro_state_boxes - boxes)
         mean_error = error.mean(dim = 0)
         print("Mean State-Space-State Error: {}ft\n".format(mean_error))
     
        
         state2 = hg.space_to_state(space_boxes)
         space2 = hg.state_to_space(state2)
         error = torch.abs(space_boxes - space2)
         mean_error = error.mean()
         print("Mean Space-State-Space Error: {}ft\n".format(mean_error))
     
         state3 = hg.space_to_state(space2)
         error = torch.abs(state3-state2)
         mean_error = error.mean(dim = 0)
         print("Mean second pass state-space-state Error: {}ft\n".format(mean_error))
         
     if True:
         print("___________________________________")
         print("Test 3: Random Box Reprojection Error (State-Im-State)")
         print("___________________________________")
         all_im_pts = torch.cat(all_im_pts,dim = 0)
         start = time.time()
         state_pts = hg.im_to_state(all_im_pts, name = all_cam_names, heights = 0,refine_heights = False)
         print("im->state took {}s for {} boxes".format(time.time()-start, all_im_pts.shape[0]))
         ### Create a random set of boxes
         boxes = torch.rand(state_pts.shape[0],6) 
         boxes[:,:2] = state_pts[:,:2]
         #boxes[:,1] = boxes[:,1] * 240 - 120
         boxes[:,2] *= 30
         boxes[:,3] *= 10
         boxes[:,4] *= 10
         boxes[:,5] = hg.polarity * torch.sign(boxes[:,1])
         
         #boxes[:,1] = 0.01
         # boxes[:,2] = 30
         # boxes[:,3:5] = .01
         
         # get appropriate camera for each object
         
         start = time.time()
         im_boxes = hg.state_to_im(boxes,name = all_cam_names)
         print("state->im took {}s for {} boxes".format(time.time()-start, all_im_pts.shape[0]))

         # plot some of the boxes
         repro_state_boxes = hg.im_to_state(im_boxes,name = all_cam_names,heights = boxes[:,4],refine_heights = True)
         
         
         error = torch.abs(repro_state_boxes - boxes)
         error = torch.nanmean(error,dim = 0)
         print("Average State-Im_State Reprojection Error: {} ft\n".format(error))
     
     
     if True:
         print("___________________________________")
         print("Test 4: Random Box Reprojection Error (Im-State-Im)")
         print("___________________________________")
         repro_im_boxes = hg.state_to_im(repro_state_boxes, name = all_cam_names)
         
         error = torch.abs(repro_im_boxes-im_boxes)
         error = torch.nanmean(error)
         print("Average Im-State-Im Reprojection Error: {} px\n".format(error))
     
     if True: 
         #plot some boxes
         for pole in [10,11]:
             for camera in range(1,7):
                     # pole = 40
                     # camera  = 2
                     plot_cam = "P{}C{}".format(str(pole).zfill(2),str(camera).zfill(2))
                     print(plot_cam)
                     im_path = os.path.join(im_dir,plot_cam) + ".png"
                     im = cv2.imread(im_path)
                     plot_boxes = []
                     plot_boxes_state = []
                     labels = []
                     for i in range(len(all_cam_names)):
                         if plot_cam in all_cam_names[i]:
                            
                             plot_boxes.append(repro_state_boxes[i])
                             plot_boxes_state.append(boxes[i])     
                             labels.append(boxes[i,5])
                     name = [plot_cam for i in range(len(plot_boxes))]
                     
                     plot_boxes_state = torch.stack(plot_boxes_state)
                     im = hg.plot_state_boxes(im,plot_boxes_state,color = (255,0,0), name = name,labels = labels)
                     
                     #plot_boxes = torch.stack(plot_boxes)
                     #im = self.plot_state_boxes(im, plot_boxes,color = (0,0,255),name = name)
                     
                     
                     
                     cv2.imshow("Frame", im)
                     cv2.waitKey(0)
                     #cv2.destroyAllWindows()
                     pass
                 
                    
 # def fill_gaps(self):
 #     additions = {}
 #     for key in self.correspondence.keys():
 #         base, direction = key.split("_")
 #         if direction == "EB":
 #             new_key = base + "_WB"
 #             if new_key not in self.correspondence.keys():
 #                 additions[new_key] = self.correspondence[key].copy()
 #         elif direction == "WB":
 #             new_key = base + "_EB"
 #             if new_key not in self.correspondence.keys():
 #                 additions[new_key] = self.correspondence[key].copy()
 #     print("Adding {} missing correspondences: {}".format(len(additions),list(additions.keys())))
 #     for key in additions.keys():

if __name__ == "__main__":
     
     save_path      = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/WACV2024_hg_save.cpkl"
     im_dir         = "/home/worklab/Documents/i24/fast-trajectory-annotator/final_dataset_preparation/4k"
     
     hg = I24_RCS(save_path = save_path,downsample = 1)
     #hg.yellow_offsets = None
     
     remove = []
     for corr in hg.correspondence.keys():
         print(corr)
         if torch.isnan(hg.correspondence[corr]["H_reference"].sum()):
             remove.append(corr)
             
     for corr in remove:
         hg.correspondence.pop(corr)
         
     print(hg.correspondence.keys())
     test_transformation(hg, im_dir)
 