# JGR-P2O

This is the Tensorflow implementation of our ECCV2020 paper "[JGR-P2O: Joint Graph Reasoning based Pixel-to-Offset Prediction Network for 3D Hand Pose Estimation from a Single Depth Image]()" 

The key ideas of JGR-P2O are two-fold: a) explicitly modeling the dependencies among joints and the relations between the pixels and the
joints for better local feature representation learning; b) unifying the
dense pixel-wise offset predictions and direct joint regression for end-toend training. 

<div align=center>
<img src="https://user-images.githubusercontent.com/22862577/87033371-b29f2800-c218-11ea-83be-0a34551c3288.png"><br>
Figure 1: Overview of a DGC layer.
</div>
Given a hand depth image, the backbone module first extracts the intermediate local feature representation X, which is then augmented by the proposed GCN-based joint graph reasoning module producing the augmented local feature representation X¯. Finally, the proposed pixel-to-offset prediction module predicts three offset maps for each joint
where each pixel value indicates the offset from the pixel to the joint along one of the axes in the UVZ coordinate system. The joint's UVZ coordinates are calculated as the
weighted average over all the pixels' predictions. Two kinds of losses, coordinate-wise regression loss Lcoordinate and pixel-wise offset regression loss Loffset, are proposed to guide the learning process. We stack together two hourglasses to enhance the learning power, feeding the output from the previous module as the input into the next while exerting intermediate supervision at the end of each module.
