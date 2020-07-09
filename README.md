# JGR-P2O

This is the Tensorflow implementation of our ECCV2020 paper "[JGR-P2O: Joint Graph Reasoning based Pixel-to-Offset Prediction Network for 3D Hand Pose Estimation from a Single Depth Image]()" 

The key ideas of JGR-P2O are two-fold: a) explicitly modeling the dependencies among joints and the relations between the pixels and the joints with  the joint graph reasoning module for better local feature representation learning; b) unifying the dense pixel-wise offset predictions and direct joint regression for end-toend training. 

<div align=center>
<img src="https://user-images.githubusercontent.com/22862577/87033371-b29f2800-c218-11ea-83be-0a34551c3288.png"><br>
Figure 1: Overview of JGR-P2O.
</div>

# Run

1. Requirements:
    * python3, tensorflow 1.12-14
    
2. datasets:
    * Download the NYU, ICVL, and MSRA datasets.
    * Download the hand centers files from [V2V-PoseNet](https://github.com/mks0601/V2V-PoseNet_RELEASE) for data preprocessing.
    * Thanks [CrossInfoNet](https://github.com/dumyy/handpose) for providing a good reference of data preprocessing.

3. training and testing:
    Here we provide an example for NYU training and testing, running on ICVL and MSRA is the same way.
    * training: python main_nyu --phase train --data_root /home/data/3D_hand_pose_estimation/NYU/dataset/
    * testing: python main_nyu --phase test --data_root /home/data/3D_hand_pose_estimation/NYU/dataset/

