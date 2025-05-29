# Video Stitcher

A Python application for stitching together video feeds from three cameras into a single panoramic video. This tool uses geometric transformations to align and blend multiple camera perspectives into a cohesive wide-angle view.

## Features

- Stitch together three separate video feeds into one continuous panoramic video
- Intuitive GUI for selecting input videos and configuring camera parameters
- Real-time preview of stitching results before processing
- Configurable camera settings (angles, field of view, camera distances)
- Preset configurations for common camera arrangements
- Support for various video formats (mp4, avi, mov, mkv)
- Customizable output resolution and location

## Requirements

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- Tkinter (for GUI)
- PIL (Python Imaging Library)
- FFmpeg (for video encoding)

## Installation

1. Clone this repository or download the source code.
2. Install the required dependencies:

```bash
pip install opencv-python numpy pillow
```

3. Ensure FFmpeg is installed on your system and available in your PATH.

## Releases

Pre-built versions of Video Stitcher are available in the Releases section of the GitHub repository:

### How to Access Releases

1. Go to the GitHub repository page
2. Click on the "Releases" tab
3. Download the latest release package

### Release Types

- **Source Code (.zip/.tar.gz)**: Complete source code that can be installed following the instructions above
- **Windows Executable (.exe)**: Standalone Windows application that doesn't require Python installation
- **Sample Videos Pack**: Collection of sample videos for testing the application, included automatically within the source code and the executable file

### Release Notes

Each release includes detailed notes about:
- New features added
- Bugs fixed
- Performance improvements
- Known issues

### Using Downloaded Releases

#### For Source Code:
Follow the installation instructions above.

#### For Windows Executable:
1. Download the .exe file
2. Run the executable directly
3. No Python installation required

## Usage

### GUI Mode

Run the application with:

```bash
python main.py
```

This will open the Video Stitcher UI where you can:

1. Select three video files (left, middle, right camera views)
2. Configure camera parameters
3. Preview the stitched result
4. Process and save the final stitched video

### Programmatic Usage

You can also use the Video Stitcher programmatically:

```python
import videoStitcher

# Create a VideoStitcher instance
stitcher = videoStitcher.VideoStitcher(
    leftVideo="path/to/left.mp4",
    middleVideo="path/to/middle.mp4",
    rightVideo="path/to/right.mp4",
    leftAngle=-30,
    rightAngle=30,
    cameraFocalHeight=1.0,
    cameraFocalLength=math.sqrt(3),
    projectionPlaneDistanceFromCenter=10,
    imageDimensions=(1920, 1080)
)

# Generate stitched video
stitcher.outputStitchedVideo("output_filename", "output_directory")
```

## Configuration Options

### Camera Setup

- **Camera Angles**: Set the inward angles of left and right cameras (typically -30° and 30°)
- **Field of View**: The camera's field of view in degrees (usually 60-70° for smartphone cameras)
- **Camera Distance**: Physical distance between cameras in centimeters
- **Image Dimensions**: Resolution of input videos

### Presets

The application includes three preset configurations:
- **Standard (30° separation)**: Common setup for most applications
- **Wide (45° separation)**: For capturing wider panoramas
- **Narrow (15° separation)**: For more detailed stitching with less perspective distortion

## Technical Details

The video stitching process involves:

1. **Transformation**: Each frame from the three cameras is transformed to align with a common projection plane
2. **Overlap Removal**: Overlapping regions between cameras are detected and cropped
3. **Stitching**: The transformed frames are combined into a single panoramic frame
4. **Video Encoding**: The stitched frames are encoded into a video file using FFmpeg

The transformation uses homography matrices to map points from the camera image planes to the projection plane, accounting for different camera angles and positions.

## Troubleshooting

- **Poor Stitching Quality**: Adjust camera angles and field of view to better match your physical camera setup
- **Processing Speed**: Lower the input video resolution for faster processing
- **Missing FFmpeg**: Ensure FFmpeg is installed correctly on your system
- **Black Regions**: Adjust the projection plane distance to reduce black regions in the output
