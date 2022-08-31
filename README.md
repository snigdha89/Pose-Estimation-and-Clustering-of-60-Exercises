# Pose-Estimation-and-Clustering-of-60-Exercises
There is a dataset of human poses while exercising in the gym. The activities are named as the first column. 
The pose-output.csv file is the result of feeding many exercise videos file into a pose estimation algorithm 
(i.e., Blazepose https://ai.googleblog.com/2020/08/on-device-real-time-body-posetracking. html). 
Each exercise is a tensor (three-dimension data, i.e., x,y,z and time) 

(1) Create a visualization that as an input we feed an exercise and creates a visualization of the physical activity as an animation or gif. 
(2) For each exercise, identify joints that are moving, and list them. Besides, identify joints that are not moving as well. 
(3) List the degree of changes between joints. For example, take a look at following the output of your algorithm should be as follows: 
{Moving: right_wrist, right_shoulder, Angle_changes: right_elbow {degree: 175, 170, … 30, 35, … 180} } 
{Not_Moving: right_hip, right_knee, right_heel, left_shoulder, left_elbow, …}
(4) Apply a clustering or classification algorithm on the result, that can
distinguish different movements or group similar ones together.
