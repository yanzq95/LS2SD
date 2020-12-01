# LS2SD
Code is modified from [fangchangma](https://github.com/fangchangma/self-supervised-depth-completion).
## testing:  
python main_test.py --evaluate your_model_path.  
  
for example:  
python main_test.py --evaluate /parent_path/2x_4x/2x/results/867.284/model_best.pth.tar  
python main_test.py --evaluate /parent_path/2x_4x/4x/results/963.634/model_best.pth.tar   

## pretrained models    
The pretrained models are available [here](https://www.dropbox.com/sh/0ow2z3grt8xb1ub/AACo0yCrmghjqgAxL4xJsuRaa?dl=0). Please put the pretrained models under '867.284/' or '963.634/' folders.  

## Comparison
Testing data examples can be found in the folder '/data/depth_selection/val_selection_cropped/'.

## prediction results  
Testing data examples can be found in folder '/data/depth_selection/val_selection_cropped/'.    
  
### 2x  
![image](https://github.com/anonymoustbd/LS2SD/blob/main/images/2x_predictions/0000000000_vis.png)    
![image](https://github.com/anonymoustbd/LS2SD/blob/main/images/2x_predictions/0000000004_vis.png)   

### 4x  
![image](https://github.com/anonymoustbd/LS2SD/blob/main/images/4x_predictions/0000000000_vis.png)    
![image](https://github.com/anonymoustbd/LS2SD/blob/main/images/4x_predictions/0000000004_vis.png)  

