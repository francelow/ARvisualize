# AR visualization with ArUco markers 
## Introduction
In this project, we will develop an application that utilizes ArUco markers to project marker-based AR visualization, image warnings, and real-time step-by-step video demonstration that will reduce or eliminate the use of paper manuals and warnings. An ArUco marker is a fiducial marker that composed with an inner binary matrix which determines its identifier and an outer black border which assists with fast detection in the image. Pose estimation is a key process of augmented reality and it is performed based on ArUco markers.For 3D visualization or image projection, we will use one ArUco marker to encode the media data, and for video renderings, we will use four ArUco markers. The four corners of each ArUco marker, called the marker coordinate system, can be used as the points to overlay the media.

### Main functionalities
Produce a chapter recap with the following information
- List of characters that appears in the various scenes of each chapter
- List of locations of the scenes in each chapter
- A plot summary of each chapter

### Pipeline
![](https://github.com/francelow/ARvisualize/blob/main/pipeline.png)  

## Dependencies

### Input Data
- ArUco markers (For more info: https://www.sciencedirect.com/science/article/abs/pii/S0031320314000235)

### Hardware
- Webcam  

### Software
- OpenCV 
- Python 3

## Demo

* Overlay an image on top of the ArUco markers  
![](https://github.com/francelow/ARvisualize/blob/main/visualize_img.gif)  
* Overlay a video on top of the ArUco markers  
![](https://github.com/francelow/ARvisualize/blob/main/visualize_video.gif)  
* Overlay a 3D model on top of the ArUco markers  
![](https://github.com/francelow/ARvisualize/blob/main/visualize_model.gif)  

## Authors
Francois Low
