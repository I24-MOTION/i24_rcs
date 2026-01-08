# i24_homography
 

## TODO
- [X] Add brief instructions
- [ ] Add description of available data files
- [X] from i24_rcs import I24_RCS (add alias in init file)
- [X] Add WSG84 -> EPSG 2274 conversion


## Usage

To install: (use v1 tag)

    conda activate <desired environment>
    pip3 install git+https://github.com/I24-MOTION/i24_rcs@<tag>

To uninstall:
  
    conda activate <desired environment>
    pip3 uninstall i24_rcs
  
To import:

    import numpy as np
    import torch
    from i24_rcs import I24_RCS
   
   
Create an object using a pre-saved data file (there are several available in this repo in the "data/" folder:

    rcs = I24_RCS(<path to homography cpkl file>)
    
    
Create some object bounding boxes on the roadway: 
    
    # Expected form is a tensor of size [n_objects,6] 
    # x_position (feet), y_position, length, width, height, direction (1 for EB or -1 for WB)
    
    road_boxes = torch.rand(10,6)
    road_boxes[:,0] *= 10000
    road_boxes[:,1] = (road_boxes[:,1] - 0.5)*48
    road_boxes[:,2] *= 20
    road_boxes[:,3] *= 8
    road_boxes[:,4] *= 7
    road_boxes[:,5] = torch.sign(road_boxes[:,1])
   
Convert from roadway coordinates (state, per internal convention) to state plane coordinates (space, per internal convention):
    
    # objects in space (state plane) are represented as tensors of shape [n_objects,n_points,3]
    # where n_points is generally either 1 for a single point or 8 for a bounding box, 
    # and the 3 values on the last dimension correspond to state plane x,y and height off roadway
    
    state_plane_boxes = rcs.state_to_space(road_boxes)
    
    
Convert from state plane coordinates to roadway coordinates:

    new_road_boxes = rcs.space_to_state(state_plane_boxes)
    
   
Compare:

    diff = torch.abs(new_road_boxes - road_boxes).sum(dim = 0)
    
In most cases, the new boxes should be within a few hundredths of a foot of the old boxes.
    
    
    
## Convert from I-24 Inception Data to GPS
Install <latest> tag of i24_rcs via: 

    source activate <desired environment to add package to>
    pip3 install git+https://github.com/I24-MOTION/i24_rcs@latest

If necessary, install pytorch.

Convert data `x_coordinates`,`y_coordinates`, each a list or array of length `N`.

     import torch
     from i24_rcs import I24_RCS

     rcs = I24_RCS( <path to rcs v1 file included in repo>)
     data = torch.zeros(N,6) # x,y,l,w,h,direction
     data[:,0] = <x_coordinates> - rcs.MM_offset
     data[:,1] = <y_coordinates> * -1
     data[:,5] = torch.sign(data[:,1]) # ensure that direction is consistent with y-coordinate sign
     state_plane_data = rcs.state_to_space(data)
     gps_data = rcs.space_to_gps(state_plane_data) 
