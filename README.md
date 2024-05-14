Localization Project
=============================

The project task is to achieve the localization for a car driving within a simulated environment, covering a minimum distance of 170m from its starting position, while keeping the distance pose error as minimum as posible, in a practical approach, this error shouldn't be bigger than 1.2 m. To accomplish this, you will make use of pointclouds extracted from a simulated car equipped with a lidar, which provides regular lidar scans. Additionally, there is an existing point cloud map "map.pcd" available, extracted from the CARLA simulator. You can achieve localization for the car by using point registration matching between the map and the scans.

We have prepared a few hands-on examples for you all to explore, which will help you grasp the concepts we've covered in the lesson more effectively.

In addition to that, we are sharing a <code>requirements.txt</code> and a [Dockerfile](./.devcontainer/build/Dockerfile) with you so you can comfortably develop your algorithms on your own computers with Anaconda, Docker Devconatiners or the environment manager of your preference. To get started, kindly refer to the [build instructions](./README.md#build-instructions) for a step-by-step guide on how to utilize them properly.

Notice that you can open all jupyter notebooks inside vscode without using the browser.

Build Instructions.
-------------------

The following instructions explain how to build the project on windows and linux in Anancoda and docker. You can also use a different environment manager if you are more familiar with it.

- [Linux Installation Instructions](./docs/Installation_linux.md)
- [Windows Installation Instructions](./docs/Installation_windows.md)

Project Instructions
--------------------

For this assignment, you will be required to complete three subtasks, which are:

1. **ICP Localization:** This involves using the Iterative Closest Points (ICP) method to perform 3D localization.
2. **NDT Localization:** This involves using the Normal Distribution Transform (NDT) algorithm to perform 3D localization.
3. **Analysis:** This task requires you to compare the results obtained from the first two tasks.

To complete each task, you have the option of using either C++ or Python. We recommend using the PCL library in C++ to carry out these tasks. 

Python examples are provided, but note that they are simplified for basic point clouds. Refer to the [Example Dataset](./Project/dataset_example.ipynb) for useful Python simplifications.

For those using C++, note that ICP and NDT are sensitive to the initial position of the vehicle. You can use the first ground truth frame for initialization. Additionally, reduce the density of the point cloud using a voxel filter, as demonstrated in the [Example Dataset](./Project/dataset_example.ipynb).

### 1 ICP Localization

In the ICP localization part of the project you will perform the localization of the car in the given map using the Iterative Closest Point algorithm. You can use the libraries provided by PCL library in c++ or Open3D in Python. This part of the project involves implementing the following series of tasks:

- Load the map from the map.pcd file.
- Iteritatively go through each frame and localize the car in the given map.
- Extract point cloud information from the dataset provided.
- Make meaningfull assumptions about pose initilization.
- Downsample the pointcloud with a **voxel grid filter**.
- Configure the ICP module to localize the vehicle.
- Use performance measures to evaluate the localization effectiviness.
- Remember that for this task, the maximum lateral error shouldn't be bigger than 1.2 meters.
- Also evaluate the computing time. There is not a harsh requirement here, but should be analized.

By completing these tasks, you'll gain valuable experience in configuring and utilizing ICP for real-world applications.

### 2 NDT Localization.

In the ICP localization part of the project you will perform the localization of the car in the given map using the Normal Distribution Transform (NDT) algorithm. You can use the libraries provided by PCL library in c++ or Open3D in Python. This part of the project involves implementing the following series of tasks:

- Load the map from the map.pcd file.
- Make meaningfull assumptions about pose initilization.
- Iteritatively go through each frame and localize the car in the given map.
- Extract point cloud information from the dataset provided.
- Downsample the pointcloud with a **voxel grid filter**.
- Configure the NDT module to localize the vehicle.
- Use performance measures to evaluate the localization effectiviness.
- Remember that for this task, you should try to achieve lateral errors below 1.2 meters.
- Also evaluate the computing time. There is not a harsh requirement here, but should be analized.

By completing these tasks, you'll gain valuable experience in configuring and utilizing NDT for real-world applications.

---

Dataset
-------

You will have to work on the dataset provided [here](https://drive.google.com/file/d/19XqBbQ6Ha3Xt_SmLVpq04Bh8UjJgxZHo/view?usp=share_link).

You will be provided with a dataset consisting of a map of a simulated environment stored in the "map.pcd" file and point clouds captured by a lidar while driving through the environment in the files "frame_\<frame_number>".

To calculate the lateral error, we have provided you with the real poses of the car through the driving in a file called "ground_truth.csv".This file has the information of the position and rotation of the car in every frame.

---

Submission Template
-------------------

The submission must be done in a zip file containing the implemented code and a report in a **PDF** with the following specifications:

### Project Overview

This section should contain a brief description of the project and what you are trying to achieve.

### Set up

This section should contain a brief description of the steps to follow to run the code. There must be a detailed explanation of how to run the implemented code so we can see run them on our local machine.

### ICP Localization

This section should describe the code utilized for ICP localization, specifically:

- Brief explanation of the ICP algorithm. 
- Module configuration and selected parameters.
- Performance metrics attained on the provided dataset.
- Runnable code that demonstrates the localization working and a visual representation of the results using a visualization library.

### Multi-Object Tracking.

This section should describe the code utilized for NDT localization, specifically:

- Brief explanation of the NDT algorithm. 
- Module configuration and selected parameters.
- Performance metrics attained on the provided dataset.
- Runnable code that demonstrates the localization working and a visual representation of the results using a visualization library.

### Analysis

This section should include a extensive analysis and comparison of both algorithms, including:

- Error through frames
- Maximum error
- Computation time
- Complexity

Please note that your report should be in PDF format, and your submission should include the implemented code and any additional materials necessary for us to replicate your results.

Good luck with your task! 🚀👍