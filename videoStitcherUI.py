import os
import math
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2

import imageTransformer
import videoStitcher
from PIL import Image, ImageTk, ImageDraw

class VideoStitcherUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Stitcher")
        # Set window to maximized state instead of fixed size
        self.root.state('zoomed')


        # Video file paths
        self.leftVideoPath = tk.StringVar()
        self.middleVideoPath = tk.StringVar()
        self.rightVideoPath = tk.StringVar()
        self.outputFolder = tk.StringVar(value="Outputs")
        self.outputFilename = tk.StringVar(value="stitched_video")

        # Camera settings
        self.leftAngle = tk.IntVar(value=-30)
        self.rightAngle = tk.IntVar(value=30)
        # User-friendly inputs
        self.fieldOfViewDegrees = tk.DoubleVar(value=60)  # Default 60 degrees
        self.cameraDistanceCm = tk.DoubleVar(value=30)  # Default 30 cm
        # Technical parameters (calculated from user-friendly inputs)
        self.cameraFocalHeight = tk.DoubleVar(value=1.0)
        self.cameraFocalLength = tk.DoubleVar(value=math.sqrt(3))
        self.projectionPlaneDistance = tk.DoubleVar(value=10)
        self.imageWidth = tk.IntVar(value=1920)
        self.imageHeight = tk.IntVar(value=1080)

        # Preset configuration
        self.presetVar = tk.StringVar(value="Standard (30° separation)")


        self._createWidgets()

    def _createWidgets(self):
        # Create main frame
        mainFrame = ttk.Frame(self.root, padding="10")
        mainFrame.pack(fill=tk.BOTH, expand=True)


        # Define help texts for camera settings - more beginner-friendly
        self.helpTexts = {
            "leftAngle": "How far the left camera is turned inward. For typical setups, use -30 degrees.",
            "rightAngle": "How far the right camera is turned inward. For typical setups, use 30 degrees.",
            "fieldOfView": "The camera's field of view in degrees. Most smartphone cameras have a FOV between 60-70 degrees.",
            "cameraDistance": "The physical distance between cameras in centimeters. Measure this on your camera setup.",
            "imageDimensions": "The dimensions of each input video (all should have the same size). Higher values give better quality but take longer to process.",
            "presets": "Quick settings for common camera arrangements. Choose the one that best matches your setup."
        }

        # File selection section
        fileFrame = ttk.LabelFrame(mainFrame, text="Video File Selection", padding="10")
        fileFrame.pack(fill=tk.X, padx=5, pady=5)

        # Add a button to load sample videos
        sampleFrame = ttk.Frame(fileFrame)
        sampleFrame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Button(sampleFrame, text="Load Sample Videos", command=self._loadSamples).pack(side=tk.LEFT, padx=5)
        ttk.Label(sampleFrame, 
                 text="(For testing - loads sample videos if available)",
                 font=("", 8, "italic")).pack(side=tk.LEFT, padx=5)

        # Left video
        ttk.Label(fileFrame, text="Left Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(fileFrame, textvariable=self.leftVideoPath, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(fileFrame, text="Browse...", command=lambda: self._browseFile(self.leftVideoPath)).grid(row=1, column=2, padx=5, pady=5)

        # Middle video
        ttk.Label(fileFrame, text="Middle Video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(fileFrame, textvariable=self.middleVideoPath, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(fileFrame, text="Browse...", command=lambda: self._browseFile(self.middleVideoPath)).grid(row=2, column=2, padx=5, pady=5)

        # Right video
        ttk.Label(fileFrame, text="Right Video:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(fileFrame, textvariable=self.rightVideoPath, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(fileFrame, text="Browse...", command=lambda: self._browseFile(self.rightVideoPath)).grid(row=3, column=2, padx=5, pady=5)

        # Camera setup diagram
        cameraDiagramFrame = ttk.LabelFrame(mainFrame, text="Camera Setup", padding="10")
        cameraDiagramFrame.pack(fill=tk.X, padx=5, pady=5)

        # Create a frame for the diagram
        diagramContainer = ttk.Frame(cameraDiagramFrame)
        diagramContainer.pack(pady=10)

        # Create a label for the diagram
        self.cameraDiagramLabel = ttk.Label(diagramContainer)
        self.cameraDiagramLabel.pack()

        # Initialize the diagram
        self._updateCameraDiagram()

        # Add a note about the diagram
        ttk.Label(cameraDiagramFrame, 
                 text="Note: The diagram shows the relative positions and angles of the three cameras.",
                 font=("", 8, "italic")).pack(pady=(0, 5))

        # Camera settings section
        settingsFrame = ttk.LabelFrame(mainFrame, text="Camera Settings", padding="10")
        settingsFrame.pack(fill=tk.X, padx=5, pady=5)

        # Add presets dropdown
        presetFrame = ttk.Frame(settingsFrame)
        presetFrame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Label(presetFrame, text="Preset Configuration:").pack(side=tk.LEFT)
        presets = ["Standard (30° separation)", "Wide (45° separation)", "Narrow (15° separation)"]
        presetDropdown = ttk.Combobox(presetFrame, textvariable=self.presetVar, values=presets, state="readonly", width=20)
        presetDropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(presetFrame, text="Apply", command=self._applyPreset).pack(side=tk.LEFT, padx=5)
        ttk.Button(presetFrame, text="?", width=2, command=lambda: self._showHelp("presets")).pack(side=tk.LEFT, padx=5)

        # Left angle
        ttk.Label(settingsFrame, text="Left Camera Angle (degrees):").grid(row=1, column=0, sticky=tk.W, pady=5)
        leftAngleSpinbox = ttk.Spinbox(settingsFrame, from_=-90, to=90, textvariable=self.leftAngle, width=10,
                                        command=self._updateCameraDiagram)
        leftAngleSpinbox.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settingsFrame, text="?", width=2, command=lambda: self._showHelp("leftAngle")).grid(row=1, column=2, padx=5, pady=5)

        # Right angle
        ttk.Label(settingsFrame, text="Right Camera Angle (degrees):").grid(row=2, column=0, sticky=tk.W, pady=5)
        rightAngleSpinbox = ttk.Spinbox(settingsFrame, from_=-90, to=90, textvariable=self.rightAngle, width=10,
                                         command=self._updateCameraDiagram)
        rightAngleSpinbox.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(settingsFrame, text="?", width=2, command=lambda: self._showHelp("rightAngle")).grid(row=2, column=2, padx=5, pady=5)

        # Field of View in degrees - User-friendly input
        ttk.Label(settingsFrame, text="Field of View (degrees):").grid(row=3, column=0, sticky=tk.W, pady=5)
        fieldOfViewSpinbox = ttk.Spinbox(settingsFrame, from_=30, to=120, increment=1, textvariable=self.fieldOfViewDegrees, width=10, 
                                           command=self._calculateTechnicalParameters)
        fieldOfViewSpinbox.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(settingsFrame, text="?", width=2, command=lambda: self._showHelp("fieldOfView")).grid(row=3, column=2, padx=5, pady=5)

        # Camera Distance in cm - User-friendly input
        ttk.Label(settingsFrame, text="Camera Distance (cm):").grid(row=4, column=0, sticky=tk.W, pady=5)
        cameraDistanceSpinbox = ttk.Spinbox(settingsFrame, from_=5, to=100, increment=1, textvariable=self.cameraDistanceCm, width=10,
                                             command=self._calculateTechnicalParameters)
        cameraDistanceSpinbox.grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(settingsFrame, text="?", width=2, command=lambda: self._showHelp("cameraDistance")).grid(row=4, column=2, padx=5, pady=5)

        # Image dimensions - Always visible
        ttk.Label(settingsFrame, text="Input Video Dimensions:").grid(row=5, column=0, sticky=tk.W, pady=5)

        dimensionsFrame = ttk.Frame(settingsFrame)
        dimensionsFrame.grid(row=5, column=1, sticky=tk.W, pady=5)

        ttk.Label(dimensionsFrame, text="Width:").pack(side=tk.LEFT)
        ttk.Spinbox(dimensionsFrame, from_=100, to=10000, textvariable=self.imageWidth, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(dimensionsFrame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(dimensionsFrame, from_=100, to=10000, textvariable=self.imageHeight, width=6).pack(side=tk.LEFT, padx=5)

        # Add help button for image dimensions
        ttk.Button(settingsFrame, text="?", width=2, command=lambda: self._showHelp("imageDimensions")).grid(row=5, column=2, sticky=tk.W, pady=5)

        # Output section
        outputFrame = ttk.LabelFrame(mainFrame, text="Output Settings", padding="10")
        outputFrame.pack(fill=tk.X, padx=5, pady=5)

        # Output folder
        ttk.Label(outputFrame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(outputFrame, textvariable=self.outputFolder, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(outputFrame, text="Browse...", command=self._browseOutputFolder).grid(row=0, column=2, padx=5, pady=5)

        # Output filename
        ttk.Label(outputFrame, text="Output Filename:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(outputFrame, textvariable=self.outputFilename, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(outputFrame, text=".mp4").grid(row=1, column=2, sticky=tk.W, pady=5)

        # Process button
        processFrame = ttk.Frame(mainFrame)
        processFrame.pack(fill=tk.X, padx=5, pady=20)

        # Add a Preview button
        ttk.Button(processFrame, text="Preview Stitch", command=self._previewStitch, width=15).pack(side=tk.RIGHT, padx=10)

        # Process Videos button with default gray color
        ttk.Button(processFrame, text="Process Videos", command=self._processVideos, width=20).pack(side=tk.RIGHT)

    def _browseFile(self, path_var):
        filetypes = (
            ('Video files', '*.mp4;*.avi;*.mov;*.mkv'),
            ('All files', '*.*')
        )

        filename = filedialog.askopenfilename(
            title='Select a video file',
            initialdir='/',
            filetypes=filetypes
        )

        if filename:
            path_var.set(filename)

    def _browseOutputFolder(self):
        folder = filedialog.askdirectory(
            title='Select output folder',
            initialdir='/'
        )

        if folder:
            self.outputFolder.set(folder)


    def _calculateTechnicalParameters(self):
        """
        Calculate technical parameters (focal height, focal length, projection plane distance)
        based on user-friendly inputs (field of view, camera distance).
        """
        # Convert field of view from degrees to radians
        fieldOfViewRadians = math.radians(self.fieldOfViewDegrees.get())

        # Calculate focal length and height to achieve the desired field of view
        # For a given FOV, the ratio of height to length is tan(FOV/2)
        # We'll keep focal_length fixed at sqrt(3) and adjust focal_height
        focalLength = math.sqrt(3)  # Fixed value for consistency
        focalHeight = focalLength * math.tan(fieldOfViewRadians / 2)

        # Calculate projection plane distance based on camera distance
        # This is a simplified relationship - we use a scaling factor to convert
        # from physical distance in cm to the internal projection plane distance
        projectionPlaneDistance = self.cameraDistanceCm.get() / 3

        # Update the technical parameters
        self.cameraFocalHeight.set(focalHeight)
        self.cameraFocalLength.set(focalLength)
        self.projectionPlaneDistance.set(projectionPlaneDistance)

    def _applyPreset(self):
        """Apply the selected preset configuration."""
        preset = self.presetVar.get()

        if preset == "Standard (30° separation)":
            self.leftAngle.set(-30)
            self.rightAngle.set(30)
            self.fieldOfViewDegrees.set(60)  # Common FOV for many cameras
            self.cameraDistanceCm.set(30)  # 30cm between cameras
        elif preset == "Wide (45° separation)":
            self.leftAngle.set(-45)
            self.rightAngle.set(45)
            self.fieldOfViewDegrees.set(70)  # Wider FOV
            self.cameraDistanceCm.set(40)  # Wider separation
        elif preset == "Narrow (15° separation)":
            self.leftAngle.set(-15)
            self.rightAngle.set(15)
            self.fieldOfViewDegrees.set(50)  # Narrower FOV
            self.cameraDistanceCm.set(20)  # Closer cameras

        # Calculate the technical parameters based on the user-friendly inputs
        self._calculateTechnicalParameters()

        # Update the camera diagram
        self._updateCameraDiagram()

    def _showHelp(self, settingKey):
        """Display help information for the specified setting."""
        helpText = self.helpTexts.get(settingKey, "No help available for this setting.")
        # Convert camelCase to Title Case for display
        displayKey = ''.join(' ' + c if c.isupper() else c for c in settingKey).strip().title()
        messagebox.showinfo(f"Help: {displayKey}", helpText)

    def _updateCameraDiagram(self):
        """Update the camera setup diagram based on current settings."""
        # Create a blank image
        width, height = 300, 150
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Calculate camera positions
        centerX, centerY = width//2, height//2
        cameraRadius = 10
        cameraDistance = 50  # Distance from center to each camera (radius of semicircle)

        # Draw the semicircle mount
        # Draw an arc from -90 to 90 degrees (top half of a circle)
        # Convert to a polygon for filling
        semicirclePoints = []
        for angle in range(-90, 91, 5):  # 5 degree steps
            angleRad = math.radians(angle)
            x = centerX + int(cameraDistance * math.cos(angleRad))
            y = centerY + int(cameraDistance * math.sin(angleRad))
            semicirclePoints.append((x, y))

        # Draw the semicircle mount
        draw.line(semicirclePoints, fill="gray", width=3)

        # Draw center camera (at 0 degrees on the semicircle)
        centerCameraX = centerX + int(cameraDistance * math.cos(0))
        centerCameraY = centerY + int(cameraDistance * math.sin(0))
        draw.ellipse([(centerCameraX - cameraRadius, centerCameraY - cameraRadius), 
                      (centerCameraX + cameraRadius, centerCameraY + cameraRadius)], 
                     outline="black", fill="lightblue")
        draw.text((centerCameraX - 15, centerCameraY + cameraRadius + 5), "Center", fill="black")

        # Draw left camera (positioned on the semicircle based on left angle)
        leftAngleRad = math.radians(self.leftAngle.get())
        leftX = centerX + int(cameraDistance * math.cos(leftAngleRad))
        leftY = centerY + int(cameraDistance * math.sin(leftAngleRad))
        draw.ellipse([(leftX - cameraRadius, leftY - cameraRadius), 
                      (leftX + cameraRadius, leftY + cameraRadius)], 
                     outline="black", fill="lightgreen")
        draw.text((leftX - 10, leftY + cameraRadius + 5), "Left", fill="black")

        # Draw right camera (positioned on the semicircle based on right angle)
        rightAngleRad = math.radians(self.rightAngle.get())
        rightX = centerX + int(cameraDistance * math.cos(rightAngleRad))
        rightY = centerY + int(cameraDistance * math.sin(rightAngleRad))
        draw.ellipse([(rightX - cameraRadius, rightY - cameraRadius), 
                      (rightX + cameraRadius, rightY + cameraRadius)], 
                     outline="black", fill="lightcoral")
        draw.text((rightX - 10, rightY + cameraRadius + 5), "Right", fill="black")

        # Draw lines showing camera viewing directions
        # Each camera points toward the center of the scene
        sceneCenterX = centerX
        sceneCenterY = centerY  # At the center of the semicircle

        # Draw the scene center point
        draw.ellipse([(sceneCenterX - 2, sceneCenterY - 2), 
                      (sceneCenterX + 2, sceneCenterY + 2)], 
                     outline="black", fill="black")

        # Draw viewing direction lines
        draw.line([(centerCameraX, centerCameraY), (sceneCenterX, sceneCenterY)], fill="blue", width=1)
        draw.line([(leftX, leftY), (sceneCenterX, sceneCenterY)], fill="green", width=1)
        draw.line([(rightX, rightY), (sceneCenterX, sceneCenterY)], fill="red", width=1)

        # Add a note about the semicircle mount
        draw.text((centerX - 70, height - 20), "Cameras mounted on semicircle", fill="black")

        # Convert to PhotoImage and update the label
        self.cameraDiagramImage = ImageTk.PhotoImage(image)
        self.cameraDiagramLabel.config(image=self.cameraDiagramImage)

    def _loadSamples(self):
        """Load sample videos if available."""
        # Check for sample videos in Test Inputs folder
        self.leftVideoPath.set("C:/Users/Lukas/CodingProjects/Camera-Dynamics/Test Inputs/left.mp4")
        self.middleVideoPath.set("C:/Users/Lukas/CodingProjects/Camera-Dynamics/Test Inputs/center.mp4")
        self.rightVideoPath.set("C:/Users/Lukas/CodingProjects/Camera-Dynamics/Test Inputs/right.mp4")
        messagebox.showinfo("Samples Loaded", "Sample videos have been loaded successfully.")
        return

    def _previewStitch(self):
        """Generate a preview of the stitched output using the first frame of each video."""
        # Ensure technical parameters are calculated from user-friendly inputs
        self._calculateTechnicalParameters()

        # Validate inputs
        if not self.leftVideoPath.get() or not self.middleVideoPath.get() or not self.rightVideoPath.get():
            messagebox.showerror("Error", "Please select all three video files.")
            return

        try:
            # Create a preview window
            previewWindow = tk.Toplevel(self.root)
            previewWindow.title("Stitching Preview")
            previewWindow.geometry("800x600")

            # Add a label explaining this is just a preview
            ttk.Label(previewWindow, text="This is a preview of how the videos will be stitched together. "
                                          "The actual output may vary slightly.",
                     font=("", 10, "italic")).pack(pady=10)

            # Try to load the first frame of each video
            leftCap = cv2.VideoCapture(self.leftVideoPath.get())
            middleCap = cv2.VideoCapture(self.middleVideoPath.get())
            rightCap = cv2.VideoCapture(self.rightVideoPath.get())

            ret1, leftFrame = leftCap.read()
            ret2, middleFrame = middleCap.read()
            ret3, rightFrame = rightCap.read()

            if not (ret1 and ret2 and ret3):
                messagebox.showerror("Error", "Could not read frames from one or more videos.")
                previewWindow.destroy()
                return

            # Create ImageTransformer with the first frames
            transformer = imageTransformer.ImageTransformer(
                leftImage=leftFrame,
                middleImage=middleFrame,
                rightImage=rightFrame,
                leftAngle=self.leftAngle.get(),
                rightAngle=self.rightAngle.get(),
                cameraFocalHeight=self.cameraFocalHeight.get(),
                cameraFocalLength=self.cameraFocalLength.get(),
                projectionPlaneDistanceFromCenter=self.projectionPlaneDistance.get(),
                imageDimensions=(self.imageWidth.get(), self.imageHeight.get())
            )

            transformationMatrices = transformer.initializeTransformationMatrices()
            transformer = imageTransformer.ImageTransformer(
                leftImage=leftFrame,
                middleImage=middleFrame,
                rightImage=rightFrame,
                leftAngle=self.leftAngle.get(),
                rightAngle=self.rightAngle.get(),
                cameraFocalHeight=self.cameraFocalHeight.get(),
                cameraFocalLength=self.cameraFocalLength.get(),
                projectionPlaneDistanceFromCenter=self.projectionPlaneDistance.get(),
                imageDimensions=(self.imageWidth.get(), self.imageHeight.get()),
                transformationMatrices=transformationMatrices
            )

            # Transform and stitch the images
            transformer.transformLeftImage()
            transformer.transformMiddleImage()
            transformer.transformRightImage()
            stitchedImage = transformer.stitchImages()

            # Convert to PIL format for display
            stitchedImageRgb = cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(stitchedImageRgb)

            # Resize if too large for display
            maxWidth = 780
            maxHeight = 500
            width, height = pilImage.size
            if width > maxWidth or height > maxHeight:
                ratio = min(maxWidth / width, maxHeight / height)
                newWidth = int(width * ratio)
                newHeight = int(height * ratio)
                pilImage = pilImage.resize((newWidth, newHeight), Image.LANCZOS)

            # Display the preview
            previewImage = ImageTk.PhotoImage(pilImage)
            previewLabel = ttk.Label(previewWindow, image=previewImage)
            previewLabel.image = previewImage  # Keep a reference
            previewLabel.pack(pady=10)

            # Add a close button
            ttk.Button(previewWindow, text="Close Preview", command=previewWindow.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Preview Error", f"An error occurred during preview generation: {str(e)}")
            if 'previewWindow' in locals():
                previewWindow.destroy()

    def _processVideos(self):
        # Ensure technical parameters are calculated from user-friendly inputs
        self._calculateTechnicalParameters()

        # Validate inputs
        if not self.leftVideoPath.get() or not self.middleVideoPath.get() or not self.rightVideoPath.get():
            messagebox.showerror("Error", "Please select all three video files.")
            return

        if not self.outputFilename.get():
            messagebox.showerror("Error", "Please enter an output filename.")
            return

        # Ensure output folder exists
        outputFolder = self.outputFolder.get()
        if not os.path.exists(outputFolder):
            try:
                os.makedirs(outputFolder)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create output folder: {str(e)}")
                return

        # Create progress window
        progressWindow = tk.Toplevel(self.root)
        progressWindow.title("Processing Videos")
        progressWindow.geometry("400x150")
        progressWindow.transient(self.root)  # Make it float on top of the main window
        progressWindow.grab_set()  # Make it modal

        # Add progress information
        ttk.Label(progressWindow, text="Processing videos. This may take several minutes...", 
                 font=("", 10, "bold")).pack(pady=(20, 10))
        progress = ttk.Progressbar(progressWindow, orient="horizontal", length=350, mode="indeterminate")
        progress.pack(pady=10, padx=20)
        progress.start()

        statusVar = tk.StringVar(value="Initializing...")
        statusLabel = ttk.Label(progressWindow, textvariable=statusVar)
        statusLabel.pack(pady=5)

        # Update the UI
        self.root.update()

        try:
            # Create VideoStitcher instance
            statusVar.set("Creating video stitcher...")
            progressWindow.update()

            videoStitcherObj = videoStitcher.VideoStitcher(
                leftVideo=self.leftVideoPath.get(),
                middleVideo=self.middleVideoPath.get(),
                rightVideo=self.rightVideoPath.get(),
                leftAngle=self.leftAngle.get(),
                rightAngle=self.rightAngle.get(),
                cameraFocalHeight=self.cameraFocalHeight.get(),
                cameraFocalLength=self.cameraFocalLength.get(),
                projectionPlaneDistanceFromCenter=self.projectionPlaneDistance.get(),
                imageDimensions=(self.imageWidth.get(), self.imageHeight.get())
            )

            # Process videos
            statusVar.set("Processing and stitching videos...")
            progressWindow.update()

            # Pass both the filename and the output directory to the VideoStitcher
            outputDir = self.outputFolder.get()
            videoStitcherObj.outputStitchedVideo(self.outputFilename.get(), outputDir)

            # Close progress window
            progressWindow.destroy()

            outputPath = os.path.join(outputDir, self.outputFilename.get() + ".mp4")
            if os.path.exists(outputPath) and os.path.getsize(outputPath) > 0:
                messagebox.showinfo("Success", f"Video processing complete. Output saved to {outputPath}")
                os.startfile(outputPath)
            else:
                messagebox.showwarning(
                    "Warning", f"Processing completed but the output file appears to be invalid or empty."
                )

        except Exception as e:
            # Close progress window
            progressWindow.destroy()
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")

def main():
    root = tk.Tk()
    app = VideoStitcherUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
