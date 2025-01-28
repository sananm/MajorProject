Smart Assistance System for the Blind using FairMOT and Depth Estimation
Overview
The Smart Assistance System for the Blind is designed to assist visually impaired individuals by detecting obstacles in their path and providing real-time alerts. Leveraging cutting-edge computer vision techniques, the system integrates FairMOT for multi-object tracking and re-identification, alongside a depth estimation model for calculating object distances from a camera feed.

Features
Multi-Object Tracking (MOT): Tracks multiple objects simultaneously in real time.
Re-Identification: Identifies the same object across different video frames.
Depth Estimation: Provides the distance of detected objects from the camera.
Real-Time Alerts: Alerts users through voice feedback if objects are too close.
System Design
The project combines the strengths of FairMOT for detection and re-identification and a monocular depth estimation model for object distance calculation.

Workflow
Capture video frames via a camera.
Pass each frame to:
FairMOT: Outputs bounding box coordinates of detected objects.
Depth Estimation Model: Outputs a depth map.
Calculate the object's distance from the camera using the depth map.
Trigger alerts if objects are within a certain threshold distance.
Technologies Used
Python 3.8
PyTorch (v1.7.x)
TensorFlow
OpenCV (v4.7.0)
CUDA Toolkit for GPU acceleration.
Models
1. FairMOT
Backbone: ResNet-34 enhanced with Deep Layer Aggregation (DLA-34).
Detection Branch: Based on CenterNet.
Re-Identification Branch: Generates 128-dimensional features for object identification.
2. Depth Estimation
Encoder-Decoder Architecture:
Encoder: DenseNet-169 pre-trained on ImageNet.
Decoder: Uses up-sampling layers with skip connections to reconstruct depth maps.
Datasets
1. CrowdHuman Dataset (for FairMOT training):
Over 15,000 images with 1.5 million human annotations.
Captures diverse poses, occlusion levels, and environments.
2. NYU Depth v2 Dataset (for depth estimation):
Includes 120k training samples and 654 testing samples.
Captures indoor environments with detailed depth maps.
Implementation
Algorithm:
Input: Video frames.
Step 1: Detect objects using FairMOT (bounding box coordinates).
Step 2: Estimate depth maps using the Depth Estimation Model.
Step 3: Calculate object distances from the depth map.
Step 4: Trigger voice alerts if objects are too close.
Results
The system was tested under various conditions, including:

Crowded streets and campuses.
Diverse lighting environments.
Key Metrics:

High object detection accuracy.
Reliable depth estimation with minimal false positives.
Real-time alerts triggered within ~60 ms/frame.
Challenges
Data Diversity: Addressing performance in crowded and dynamic environments.
Real-Time Processing: Ensuring minimal latency with large input frames.
Future Work
Android Integration: Developing a smartphone app for accessibility.
Sensor Integration: Exploring LiDAR for enhanced depth accuracy.
Class Expansion: Training models on datasets with diverse object classes for broader usability.
Installation and Usage
Prerequisites
Python 3.8+
PyTorch, TensorFlow, and OpenCV installed.
A GPU with CUDA support (optional but recommended).
Steps:
Clone the repository.
bash
Copy
Edit
git clone https://github.com/your-repo-link.git
cd your-repo-folder
Install dependencies.
bash
Copy
Edit
pip install -r requirements.txt
Download pretrained models for:
FairMOT: [Link to weights]
Depth Estimation: [Link to weights]
Run the system:
bash
Copy
Edit
python main.py --input_video path_to_video
References
Zhang, Yifu et al., "FairMOT: On the fairness of detection and re-identification in multiple object tracking," IJCV, 2021.
Alhashim, I., & Wonka, P., "High Quality Monocular Depth Estimation via Transfer Learning," arXiv, 2018.
Authors
Ananya Vudumula
Mohammed Sanan Moinuddin
Supervisor: Dr. M. Swamy Das, Chaitanya Bharathi Institute of Technology.
