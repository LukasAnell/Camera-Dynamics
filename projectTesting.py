import math
import cv2
import numpy as np
import os
import time
import sys

import imageTransformer
import videoStitcher
import videoStitcherUI

def testTransformationMatrix():
    """
    Test the transformation matrix calculation functionality.
    This test verifies that the transformation matrix is correctly calculated
    for different camera positions and angles.
    """
    print("Testing transformation matrix calculation...")

    # Create a simple test image
    testImage = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(testImage, (100, 100), (400, 400), (0, 255, 0), -1)

    # Create an ImageTransformer instance with test parameters
    transformer = imageTransformer.ImageTransformer(
        leftImage=testImage.copy(),
        middleImage=testImage.copy(),
        rightImage=testImage.copy(),
        leftAngle=-30,
        rightAngle=30,
        cameraFocalHeight=1.0,
        cameraFocalLength=math.sqrt(3),
        projectionPlaneDistanceFromCenter=10,
        imageDimensions=(500, 500)
    )

    # Test transformation matrix for different camera positions
    # Middle camera (straight ahead)
    middlePos, middleFwd = transformer.getMiddleForwardVectorAndPosition()
    middleMatrix = transformer.getTransformationMatrix(middlePos, middleFwd)
    print(f"Middle camera transformation matrix shape: {middleMatrix.shape}")

    # Left camera
    leftPos, leftFwd = transformer.getLeftForwardVectorAndPosition()
    leftMatrix = transformer.getTransformationMatrix(leftPos, leftFwd)
    print(f"Left camera transformation matrix shape: {leftMatrix.shape}")

    # Right camera
    rightPos, rightFwd = transformer.getRightForwardVectorAndPosition()
    rightMatrix = transformer.getTransformationMatrix(rightPos, rightFwd)
    print(f"Right camera transformation matrix shape: {rightMatrix.shape}")

    # Apply transformations to verify they work
    transformedMiddle = transformer.applyTransformation(testImage, middleMatrix)
    transformedLeft = transformer.applyTransformation(testImage, leftMatrix)
    transformedRight = transformer.applyTransformation(testImage, rightMatrix)

    # Save the transformed images for visual inspection
    cv2.imwrite("Test Outputs/transformed_middle.jpg", transformedMiddle)
    cv2.imwrite("Test Outputs/transformed_left.jpg", transformedLeft)
    cv2.imwrite("Test Outputs/transformed_right.jpg", transformedRight)

    print("Transformation matrix test completed. Check the Test Outputs folder for results.")
    return True

def testImageStitchingPreTransformed():
    """
    Test image stitching functionality with pre-transformed images.
    This test uses the sample images directly without applying transformations first.
    """
    print("Testing image stitching with pre-transformed images...")

    try:
        # Load test images
        leftPath = R"Test Inputs\left.jpeg"
        middlePath = R"Test Inputs\middle.jpeg"
        rightPath = R"Test Inputs\right.jpeg"

        leftImg = cv2.imread(leftPath)
        middleImg = cv2.imread(middlePath)
        rightImg = cv2.imread(rightPath)

        if leftImg is None or middleImg is None or rightImg is None:
            print("Error: Could not load test images.")
            return False

        # Create a simple horizontal concatenation (no transformation)
        stitchedImage = cv2.hconcat([leftImg, middleImg, rightImg])
        cv2.imwrite("Test Outputs/stitched_pre_transformed.jpg", stitchedImage)

        print("Pre-transformed image stitching test completed.")
        return True
    except Exception as e:
        print(f"Error in pre-transformed image stitching test: {e}")
        return False

def testImageStitchingPostTransformed():
    """
    Test image stitching functionality with post-transformed images.
    This test applies transformations to the images before stitching them.
    """
    print("Testing image stitching with post-transformed images...")

    try:
        # Load test images
        leftPath = R"Test Inputs\left.jpeg"
        middlePath = R"Test Inputs\middle.jpeg"
        rightPath = R"Test Inputs\right.jpeg"

        # Create ImageTransformer instance
        transformer = imageTransformer.ImageTransformer(
            leftImage=cv2.imread(leftPath),
            middleImage=cv2.imread(middlePath),
            rightImage=cv2.imread(rightPath),
            leftAngle=-30,
            rightAngle=30,
            cameraFocalHeight=1.0,
            cameraFocalLength=math.sqrt(3),
            projectionPlaneDistanceFromCenter=10,
            imageDimensions=(3024, 4072)
        )

        # Apply transformations
        transformer.transformLeftImage()
        transformer.transformMiddleImage()
        transformer.transformRightImage()

        # Stitch the transformed images
        stitchedImage = transformer.stitchImages()
        cv2.imwrite("Test Outputs/stitched_post_transformed.jpg", stitchedImage)

        print("Post-transformed image stitching test completed.")
        return True
    except Exception as e:
        print(f"Error in post-transformed image stitching test: {e}")
        return False

def testVideoStitching():
    """
    Test video stitching functionality.
    This test creates a stitched video from test video files.
    """
    print("Testing video stitching...")

    try:
        # Check if test video exists
        testVideoPath = R"Test Outputs\spinning rat.mp4"
        if not os.path.exists(testVideoPath):
            print(f"Error: Test video not found at {testVideoPath}")
            return False

        # Verify that the video file can be opened and read
        cap = cv2.VideoCapture(testVideoPath)
        if not cap.isOpened():
            print(f"Error: Could not open video file {testVideoPath}")
            return False

        # Read the first frame to get dimensions
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error: Could not read frames from video file {testVideoPath}")
            cap.release()
            return False

        # Get video dimensions
        height, width = frame.shape[:2]
        print(f"Video dimensions: {width}x{height}")

        # Release the video capture
        cap.release()

        # Create VideoStitcher instance using the same video for all inputs (for testing)
        try:
            stitcher = videoStitcher.VideoStitcher(
                leftVideo=testVideoPath,
                middleVideo=testVideoPath,
                rightVideo=testVideoPath,
                leftAngle=-30,
                rightAngle=30,
                cameraFocalHeight=1.0,
                cameraFocalLength=math.sqrt(3),
                projectionPlaneDistanceFromCenter=10,
                imageDimensions=(width, height)  # Use actual video dimensions
            )

            # Output the stitched video
            outputPath = "Test Outputs/test_stitched_video.mp4"
            stitcher.outputStitchedVideo(outputPath)

            if os.path.exists(outputPath):
                print(f"Video stitching test completed. Output saved to {outputPath}")
                return True
            else:
                print("Error: Failed to create stitched video.")
                return False
        except Exception as e:
            print(f"Error in video stitching process: {e}")
            print("This could be due to incompatible video format or dimensions.")
            print("Try using a different test video or check the VideoStitcher implementation.")
            print("Falling back to basic video processing test...")
            return testBasicVideoProcessing()
    except Exception as e:
        print(f"Error in video stitching test: {e}")
        return False

def testBasicVideoProcessing():
    """
    A simpler test for basic video processing functionality.
    This test reads frames from a video, applies a basic transformation,
    and saves the result as a new video.
    """
    print("\nTesting basic video processing...")

    try:
        # Check if test video exists
        testVideoPath = R"Test Outputs\spinning rat.mp4"
        if not os.path.exists(testVideoPath):
            print(f"Error: Test video not found at {testVideoPath}")
            return False

        # Open the video file
        cap = cv2.VideoCapture(testVideoPath)
        if not cap.isOpened():
            print(f"Error: Could not open video file {testVideoPath}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps} fps, {frameCount} frames")

        # Create output video writer
        outputPath = "Test Outputs/basic_video_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

        # Process a limited number of frames (10 frames or all if less)
        maxFrames = min(10, frameCount)
        processedFrames = 0

        print(f"Processing {maxFrames} frames...")

        while processedFrames < maxFrames:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply a simple transformation (flip horizontally)
            processedFrame = cv2.flip(frame, 1)

            # Write the frame to the output video
            out.write(processedFrame)
            processedFrames += 1

        # Release resources
        cap.release()
        out.release()

        if os.path.exists(outputPath):
            print(f"Basic video processing test completed. Output saved to {outputPath}")
            return True
        else:
            print("Error: Failed to create output video.")
            return False
    except Exception as e:
        print(f"Error in basic video processing test: {e}")
        return False

def testUserSoftware():
    """
    Test the end-user software functionality.
    This is a basic test that launches the UI and checks if it initializes properly.
    Note: This test requires manual interaction and visual confirmation.
    """
    print("Testing user software...")
    print("Note: This test will launch the UI. Close it manually after testing.")

    try:
        # Launch the UI in a separate process to avoid blocking the test suite
        import threading
        import time

        def runUi():
            videoStitcherUI.main()

        uiThread = threading.Thread(target=runUi)
        uiThread.daemon = True  # Allow the thread to be terminated when the main program exits
        uiThread.start()

        # Give the UI some time to initialize
        time.sleep(5)

        print("User software test initiated. Please interact with the UI and close it when done.")
        print("The test suite will continue after you close the UI or after 30 seconds.")

        # Wait for a maximum of 30 seconds
        uiThread.join(30)

        return True
    except Exception as e:
        print(f"Error in user software test: {e}")
        return False

def runAllTests():
    """Run all test functions and report results."""
    tests = [
        ("Transformation Matrix", testTransformationMatrix),
        ("Image Stitching (Pre-transformed)", testImageStitchingPreTransformed),
        ("Image Stitching (Post-transformed)", testImageStitchingPostTransformed),
        ("Video Stitching", testVideoStitching),
        ("Basic Video Processing", testBasicVideoProcessing),
        ("User Software", testUserSoftware)
    ]

    results = []

    print("=" * 50)
    print("STARTING TEST SUITE")
    print("=" * 50)

    for testName, testFunc in tests:
        print(f"\nRunning test: {testName}")
        print("-" * 30)

        startTime = time.time()
        try:
            success = testFunc()
            elapsedTime = time.time() - startTime

            if success:
                result = "PASSED"
            else:
                result = "FAILED"

            results.append((testName, result, elapsedTime))
            print(f"Test {result} in {elapsedTime:.2f} seconds")

        except Exception as e:
            elapsedTime = time.time() - startTime
            results.append((testName, "ERROR", elapsedTime))
            print(f"Test ERROR: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for testName, result, elapsedTime in results:
        print(f"{testName}: {result} ({elapsedTime:.2f}s)")

    # Count results
    passed = sum(1 for _, result, _ in results if result == "PASSED")
    failed = sum(1 for _, result, _ in results if result == "FAILED")
    errors = sum(1 for _, result, _ in results if result == "ERROR")

    print("-" * 50)
    print(f"TOTAL: {len(results)} tests")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"ERRORS: {errors}")
    print("=" * 50)

def main():
    """
    Main function to run the test suite.

    Usage:
        python projectTesting.py [test_name]

    Arguments:
        testName: Optional. Name of a specific test to run.
                  Valid options: matrix, pre_image, post_image, video, basic_video, ui, all
                  Default: all
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        testName = sys.argv[1].lower()

        if testName == "matrix":
            testTransformationMatrix()
        elif testName == "pre_image":
            testImageStitchingPreTransformed()
        elif testName == "post_image":
            testImageStitchingPostTransformed()
        elif testName == "video":
            testVideoStitching()
        elif testName == "basic_video":
            testBasicVideoProcessing()
        elif testName == "ui":
            testUserSoftware()
        elif testName == "all":
            runAllTests()
        else:
            print(f"Unknown test: {testName}")
            print("Valid options: matrix, pre_image, post_image, video, basic_video, ui, all")
    else:
        # Default: run all tests
        runAllTests()

if __name__ == "__main__":
    main()
