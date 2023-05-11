# i24_homography


## TODO
- [X] Add brief instructions
- [ ] Add description of available data files
- [X] from i24_rcs import I24_RCS (add alias in init file)
- [ ] Add WSG84 -> EPSG 2274 conversion


## Usage

To install: (use v1 tag)

    conda activate <desired environment>
    pip3 install git+https://github.com/DerekGloudemans/i24_rcs@<tag>

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
    
    # x_position, y_position, length, width, height, direction
    road_boxes = torch.rand(10,6)
    road_boxes[:,0] *= 10000
    road_boxes[:,1] = (road_boxes[:,1] - 0.5)*48
    road_boxes[:,2] *= 20
    road_boxes[:,3] *= 8
    road_boxes[:,4] *= 7
    road_boxes[:,5] = torch.sign(road_boxes[:,1])
   
Convert from roadway coordinates (state, per internal convention) to state plane coordinates (space, per internal convention):
   
    state_plane_boxes = rcs.state_to_space(road_boxes)
    
    
Convert from state plane coordinates to roadway coordinates:

    new_road_boxes = rcs.space_to_state(state_plane_boxes)
    
   
Compare:

    diff = torch.abs(new_road_boxes - road_boxes).sum(dim = 0)
    
In most cases, the new boxes should be within a few hundredths of a foot of the old boxes.
    
    
    
