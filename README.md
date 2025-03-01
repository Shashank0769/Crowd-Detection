# Crowd Detection Using OpenCV

## Overview

This project implements a crowd detection system using the YOLOv3 (You Only Look Once) object detection model. The system processes a video file to detect persons and identifies crowd events based on specific criteria.

## Features

- Detects persons in video frames using YOLOv3.
- Identifies crowd events defined as three or more persons standing close together for at least 10 consecutive frames.
- Logs detected crowd events, including frame numbers and the count of persons in the crowd.
- Saves the results in a CSV file for further analysis.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas
- yolov3.weights
- yolov3.cfg

You can install the required packages using pip:

```bash
pip install opencv-python opencv-python-headless numpy pandas
```

## Setup

### Clone the Repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### Download YOLOv3 Weights and Configuration:

Download the YOLOv3 weights and configuration files from the following links:

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

Place the downloaded files in the project directory.

### Run the Script:

Modify the `video_path` variable in the `detection.py` script to point to your input video file.

Execute the script:

```bash
python detection.py
```

## Output

The results of the crowd detection will be saved in a CSV file named `crowd_detection_results.csv`, which will contain the following columns:

- **Frame Number:** The frame number where a crowd was detected.
- **Person Count in Crowd:** The number of persons detected in the crowd.

## Example

To run the project, you can use a sample video file. Make sure to update the `video_path` variable in the `detection.py` script to point to your video file.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv3 for the object detection model.
- OpenCV for computer vision tasks.

## Preview-Video Sample

![Screenshot 2025-03-01 141333](https://github.com/user-attachments/assets/94599a90-95b5-4613-8188-850e1e9f10b8)

## CSV Output

![Screenshot (428)](https://github.com/user-attachments/assets/fceb2312-5e17-4adf-a25c-ab194ee6ac03)

