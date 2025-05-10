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
        self.left_video_path = tk.StringVar()
        self.middle_video_path = tk.StringVar()
        self.right_video_path = tk.StringVar()
        self.output_folder = tk.StringVar(value="Outputs")
        self.output_filename = tk.StringVar(value="stitched_video")

        # Camera settings
        self.left_angle = tk.IntVar(value=-30)
        self.right_angle = tk.IntVar(value=30)
        # User-friendly inputs
        self.field_of_view_degrees = tk.DoubleVar(value=60)  # Default 60 degrees
        self.camera_distance_cm = tk.DoubleVar(value=30)  # Default 30 cm
        # Technical parameters (calculated from user-friendly inputs)
        self.camera_focal_height = tk.DoubleVar(value=1.0)
        self.camera_focal_length = tk.DoubleVar(value=math.sqrt(3))
        self.projection_plane_distance = tk.DoubleVar(value=10)
        self.image_width = tk.IntVar(value=3024)
        self.image_height = tk.IntVar(value=4072)

        # Preset configuration
        self.preset_var = tk.StringVar(value="Standard (30° separation)")


        self._create_widgets()

    def _create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)


        # Define help texts for camera settings - more beginner-friendly
        self.help_texts = {
            "left_angle": "How far the left camera is turned inward. For typical setups, use -30 degrees.",
            "right_angle": "How far the right camera is turned inward. For typical setups, use 30 degrees.",
            "field_of_view": "The camera's field of view in degrees. Most smartphone cameras have a FOV between 60-70 degrees.",
            "camera_distance": "The physical distance between cameras in centimeters. Measure this on your camera setup.",
            "image_dimensions": "The dimensions of each input video (all should have the same size). Higher values give better quality but take longer to process.",
            "presets": "Quick settings for common camera arrangements. Choose the one that best matches your setup."
        }

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Video File Selection", padding="10")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add a button to load sample videos
        sample_frame = ttk.Frame(file_frame)
        sample_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Button(sample_frame, text="Load Sample Videos", command=self._load_samples).pack(side=tk.LEFT, padx=5)
        ttk.Label(sample_frame, 
                 text="(For testing - loads sample videos if available)",
                 font=("", 8, "italic")).pack(side=tk.LEFT, padx=5)

        # Left video
        ttk.Label(file_frame, text="Left Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.left_video_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=lambda: self._browse_file(self.left_video_path)).grid(row=1, column=2, padx=5, pady=5)

        # Middle video
        ttk.Label(file_frame, text="Middle Video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.middle_video_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=lambda: self._browse_file(self.middle_video_path)).grid(row=2, column=2, padx=5, pady=5)

        # Right video
        ttk.Label(file_frame, text="Right Video:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.right_video_path, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=lambda: self._browse_file(self.right_video_path)).grid(row=3, column=2, padx=5, pady=5)

        # Camera setup diagram
        camera_diagram_frame = ttk.LabelFrame(main_frame, text="Camera Setup", padding="10")
        camera_diagram_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create a frame for the diagram
        diagram_container = ttk.Frame(camera_diagram_frame)
        diagram_container.pack(pady=10)

        # Create a label for the diagram
        self.camera_diagram_label = ttk.Label(diagram_container)
        self.camera_diagram_label.pack()

        # Initialize the diagram
        self._update_camera_diagram()

        # Add a note about the diagram
        ttk.Label(camera_diagram_frame, 
                 text="Note: The diagram shows the relative positions and angles of the three cameras.",
                 font=("", 8, "italic")).pack(pady=(0, 5))

        # Camera settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Camera Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add presets dropdown
        preset_frame = ttk.Frame(settings_frame)
        preset_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Label(preset_frame, text="Preset Configuration:").pack(side=tk.LEFT)
        presets = ["Standard (30° separation)", "Wide (45° separation)", "Narrow (15° separation)"]
        preset_dropdown = ttk.Combobox(preset_frame, textvariable=self.preset_var, values=presets, state="readonly", width=20)
        preset_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Apply", command=self._apply_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="?", width=2, command=lambda: self._show_help("presets")).pack(side=tk.LEFT, padx=5)

        # Left angle
        ttk.Label(settings_frame, text="Left Camera Angle (degrees):").grid(row=1, column=0, sticky=tk.W, pady=5)
        left_angle_spinbox = ttk.Spinbox(settings_frame, from_=-90, to=90, textvariable=self.left_angle, width=10,
                                        command=self._update_camera_diagram)
        left_angle_spinbox.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("left_angle")).grid(row=1, column=2, padx=5, pady=5)

        # Right angle
        ttk.Label(settings_frame, text="Right Camera Angle (degrees):").grid(row=2, column=0, sticky=tk.W, pady=5)
        right_angle_spinbox = ttk.Spinbox(settings_frame, from_=-90, to=90, textvariable=self.right_angle, width=10,
                                         command=self._update_camera_diagram)
        right_angle_spinbox.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("right_angle")).grid(row=2, column=2, padx=5, pady=5)

        # Field of View in degrees - User-friendly input
        ttk.Label(settings_frame, text="Field of View (degrees):").grid(row=3, column=0, sticky=tk.W, pady=5)
        field_of_view_spinbox = ttk.Spinbox(settings_frame, from_=30, to=120, increment=1, textvariable=self.field_of_view_degrees, width=10, 
                                           command=self._calculate_technical_parameters)
        field_of_view_spinbox.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("field_of_view")).grid(row=3, column=2, padx=5, pady=5)

        # Camera Distance in cm - User-friendly input
        ttk.Label(settings_frame, text="Camera Distance (cm):").grid(row=4, column=0, sticky=tk.W, pady=5)
        camera_distance_spinbox = ttk.Spinbox(settings_frame, from_=5, to=100, increment=1, textvariable=self.camera_distance_cm, width=10,
                                             command=self._calculate_technical_parameters)
        camera_distance_spinbox.grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("camera_distance")).grid(row=4, column=2, padx=5, pady=5)

        # Image dimensions - Always visible
        ttk.Label(settings_frame, text="Input Video Dimensions:").grid(row=5, column=0, sticky=tk.W, pady=5)

        dimensions_frame = ttk.Frame(settings_frame)
        dimensions_frame.grid(row=5, column=1, sticky=tk.W, pady=5)

        ttk.Label(dimensions_frame, text="Width:").pack(side=tk.LEFT)
        ttk.Spinbox(dimensions_frame, from_=100, to=10000, textvariable=self.image_width, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(dimensions_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(dimensions_frame, from_=100, to=10000, textvariable=self.image_height, width=6).pack(side=tk.LEFT, padx=5)

        # Add help button for image dimensions
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("image_dimensions")).grid(row=5, column=2, sticky=tk.W, pady=5)

        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        # Output folder
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse...", command=self._browse_output_folder).grid(row=0, column=2, padx=5, pady=5)

        # Output filename
        ttk.Label(output_frame, text="Output Filename:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_filename, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(output_frame, text=".mp4").grid(row=1, column=2, sticky=tk.W, pady=5)

        # Process button
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, padx=5, pady=20)

        # Add a Preview button
        ttk.Button(process_frame, text="Preview Stitch", command=self._preview_stitch, width=15).pack(side=tk.RIGHT, padx=10)

        # Process Videos button with default gray color
        ttk.Button(process_frame, text="Process Videos", command=self._process_videos, width=20).pack(side=tk.RIGHT)

    def _browse_file(self, path_var):
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

    def _browse_output_folder(self):
        folder = filedialog.askdirectory(
            title='Select output folder',
            initialdir='/'
        )

        if folder:
            self.output_folder.set(folder)


    def _calculate_technical_parameters(self):
        """
        Calculate technical parameters (focal height, focal length, projection plane distance)
        based on user-friendly inputs (field of view, camera distance).
        """
        # Convert field of view from degrees to radians
        field_of_view_radians = math.radians(self.field_of_view_degrees.get())

        # Calculate focal length and height to achieve the desired field of view
        # For a given FOV, the ratio of height to length is tan(FOV/2)
        # We'll keep focal_length fixed at sqrt(3) and adjust focal_height
        focal_length = math.sqrt(3)  # Fixed value for consistency
        focal_height = focal_length * math.tan(field_of_view_radians / 2)

        # Calculate projection plane distance based on camera distance
        # This is a simplified relationship - we use a scaling factor to convert
        # from physical distance in cm to the internal projection plane distance
        projection_plane_distance = self.camera_distance_cm.get() / 3

        # Update the technical parameters
        self.camera_focal_height.set(focal_height)
        self.camera_focal_length.set(focal_length)
        self.projection_plane_distance.set(projection_plane_distance)

    def _apply_preset(self):
        """Apply the selected preset configuration."""
        preset = self.preset_var.get()

        if preset == "Standard (30° separation)":
            self.left_angle.set(-30)
            self.right_angle.set(30)
            self.field_of_view_degrees.set(60)  # Common FOV for many cameras
            self.camera_distance_cm.set(30)  # 30cm between cameras
        elif preset == "Wide (45° separation)":
            self.left_angle.set(-45)
            self.right_angle.set(45)
            self.field_of_view_degrees.set(70)  # Wider FOV
            self.camera_distance_cm.set(40)  # Wider separation
        elif preset == "Narrow (15° separation)":
            self.left_angle.set(-15)
            self.right_angle.set(15)
            self.field_of_view_degrees.set(50)  # Narrower FOV
            self.camera_distance_cm.set(20)  # Closer cameras

        # Calculate the technical parameters based on the user-friendly inputs
        self._calculate_technical_parameters()

        # Update the camera diagram
        self._update_camera_diagram()

    def _show_help(self, setting_key):
        """Display help information for the specified setting."""
        help_text = self.help_texts.get(setting_key, "No help available for this setting.")
        messagebox.showinfo(f"Help: {setting_key.replace('_', ' ').title()}", help_text)

    def _update_camera_diagram(self):
        """Update the camera setup diagram based on current settings."""
        # Create a blank image
        width, height = 300, 150
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Calculate camera positions
        center_x, center_y = width//2, height//2
        camera_radius = 10
        camera_distance = 50  # Distance from center to each camera (radius of semicircle)

        # Draw the semicircle mount
        # Draw an arc from -90 to 90 degrees (top half of a circle)
        # Convert to a polygon for filling
        semicircle_points = []
        for angle in range(-90, 91, 5):  # 5 degree steps
            angle_rad = math.radians(angle)
            x = center_x + int(camera_distance * math.cos(angle_rad))
            y = center_y + int(camera_distance * math.sin(angle_rad))
            semicircle_points.append((x, y))

        # Draw the semicircle mount
        draw.line(semicircle_points, fill="gray", width=3)

        # Draw center camera (at 0 degrees on the semicircle)
        center_camera_x = center_x + int(camera_distance * math.cos(0))
        center_camera_y = center_y + int(camera_distance * math.sin(0))
        draw.ellipse([(center_camera_x - camera_radius, center_camera_y - camera_radius), 
                      (center_camera_x + camera_radius, center_camera_y + camera_radius)], 
                     outline="black", fill="lightblue")
        draw.text((center_camera_x - 15, center_camera_y + camera_radius + 5), "Center", fill="black")

        # Draw left camera (positioned on the semicircle based on left angle)
        left_angle_rad = math.radians(self.left_angle.get())
        left_x = center_x + int(camera_distance * math.cos(left_angle_rad))
        left_y = center_y + int(camera_distance * math.sin(left_angle_rad))
        draw.ellipse([(left_x - camera_radius, left_y - camera_radius), 
                      (left_x + camera_radius, left_y + camera_radius)], 
                     outline="black", fill="lightgreen")
        draw.text((left_x - 10, left_y + camera_radius + 5), "Left", fill="black")

        # Draw right camera (positioned on the semicircle based on right angle)
        right_angle_rad = math.radians(self.right_angle.get())
        right_x = center_x + int(camera_distance * math.cos(right_angle_rad))
        right_y = center_y + int(camera_distance * math.sin(right_angle_rad))
        draw.ellipse([(right_x - camera_radius, right_y - camera_radius), 
                      (right_x + camera_radius, right_y + camera_radius)], 
                     outline="black", fill="lightcoral")
        draw.text((right_x - 10, right_y + camera_radius + 5), "Right", fill="black")

        # Draw lines showing camera viewing directions
        # Each camera points toward the center of the scene
        scene_center_x = center_x
        scene_center_y = center_y  # At the center of the semicircle

        # Draw the scene center point
        draw.ellipse([(scene_center_x - 2, scene_center_y - 2), 
                      (scene_center_x + 2, scene_center_y + 2)], 
                     outline="black", fill="black")

        # Draw viewing direction lines
        draw.line([(center_camera_x, center_camera_y), (scene_center_x, scene_center_y)], fill="blue", width=1)
        draw.line([(left_x, left_y), (scene_center_x, scene_center_y)], fill="green", width=1)
        draw.line([(right_x, right_y), (scene_center_x, scene_center_y)], fill="red", width=1)

        # Add a note about the semicircle mount
        draw.text((center_x - 70, height - 20), "Cameras mounted on semicircle", fill="black")

        # Convert to PhotoImage and update the label
        self.camera_diagram_image = ImageTk.PhotoImage(image)
        self.camera_diagram_label.config(image=self.camera_diagram_image)

    def _load_samples(self):
        """Load sample videos if available."""
        # Check for sample videos in Test Inputs folder
        sample_dir = "Test Inputs"
        if os.path.exists(sample_dir):
            # Look for video files
            video_files = []
            for file in os.listdir(sample_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(sample_dir, file))
                elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # For testing, we can also use image files
                    video_files.append(os.path.join(sample_dir, file))

            # If we found at least 3 files, use them
            if len(video_files) >= 3:
                self.left_video_path.set(video_files[0])
                self.middle_video_path.set(video_files[1])
                self.right_video_path.set(video_files[2])
                messagebox.showinfo("Samples Loaded", "Sample videos have been loaded successfully.")
                return

        # If we get here, we couldn't find sample videos
        messagebox.showinfo("No Samples Found", 
                           "No sample videos were found in the 'Test Inputs' folder. "
                           "Please create this folder and add at least 3 video files to use this feature.")

    def _preview_stitch(self):
        """Generate a preview of the stitched output using the first frame of each video."""
        # Ensure technical parameters are calculated from user-friendly inputs
        self._calculate_technical_parameters()

        # Validate inputs
        if not self.left_video_path.get() or not self.middle_video_path.get() or not self.right_video_path.get():
            messagebox.showerror("Error", "Please select all three video files.")
            return

        try:
            # Create a preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Stitching Preview")
            preview_window.geometry("800x600")

            # Add a label explaining this is just a preview
            ttk.Label(preview_window, text="This is a preview of how the videos will be stitched together. "
                                          "The actual output may vary slightly.",
                     font=("", 10, "italic")).pack(pady=10)

            # Try to load the first frame of each video
            left_cap = cv2.VideoCapture(self.left_video_path.get())
            middle_cap = cv2.VideoCapture(self.middle_video_path.get())
            right_cap = cv2.VideoCapture(self.right_video_path.get())

            ret1, left_frame = left_cap.read()
            ret2, middle_frame = middle_cap.read()
            ret3, right_frame = right_cap.read()

            if not (ret1 and ret2 and ret3):
                messagebox.showerror("Error", "Could not read frames from one or more videos.")
                preview_window.destroy()
                return

            # Create ImageTransformer with the first frames
            transformer = imageTransformer.ImageTransformer(
                leftImage=left_frame,
                middleImage=middle_frame,
                rightImage=right_frame,
                leftAngle=self.left_angle.get(),
                rightAngle=self.right_angle.get(),
                cameraFocalHeight=self.camera_focal_height.get(),
                cameraFocalLength=self.camera_focal_length.get(),
                projectionPlaneDistanceFromCenter=self.projection_plane_distance.get(),
                imageDimensions=(self.image_width.get(), self.image_height.get())
            )

            # Transform and stitch the images
            transformer.transformLeftImage()
            transformer.transformMiddleImage()
            transformer.transformRightImage()
            stitched_image = transformer.stitchImages()

            # Convert to PIL format for display
            stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(stitched_image_rgb)

            # Resize if too large for display
            max_width = 780
            max_height = 500
            width, height = pil_image.size
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

            # Display the preview
            preview_image = ImageTk.PhotoImage(pil_image)
            preview_label = ttk.Label(preview_window, image=preview_image)
            preview_label.image = preview_image  # Keep a reference
            preview_label.pack(pady=10)

            # Add a close button
            ttk.Button(preview_window, text="Close Preview", command=preview_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Preview Error", f"An error occurred during preview generation: {str(e)}")
            if 'preview_window' in locals():
                preview_window.destroy()

    def _process_videos(self):
        # Ensure technical parameters are calculated from user-friendly inputs
        self._calculate_technical_parameters()

        # Validate inputs
        if not self.left_video_path.get() or not self.middle_video_path.get() or not self.right_video_path.get():
            messagebox.showerror("Error", "Please select all three video files.")
            return

        if not self.output_filename.get():
            messagebox.showerror("Error", "Please enter an output filename.")
            return

        # Ensure output folder exists
        output_folder = self.output_folder.get()
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create output folder: {str(e)}")
                return

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Videos")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)  # Make it float on top of the main window
        progress_window.grab_set()  # Make it modal

        # Add progress information
        ttk.Label(progress_window, text="Processing videos. This may take several minutes...", 
                 font=("", 10, "bold")).pack(pady=(20, 10))
        progress = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="indeterminate")
        progress.pack(pady=10, padx=20)
        progress.start()

        status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(pady=5)

        # Update the UI
        self.root.update()

        try:
            # Create VideoStitcher instance
            status_var.set("Creating video stitcher...")
            progress_window.update()

            video_stitcher = videoStitcher.VideoStitcher(
                leftVideo=self.left_video_path.get(),
                middleVideo=self.middle_video_path.get(),
                rightVideo=self.right_video_path.get(),
                leftAngle=self.left_angle.get(),
                rightAngle=self.right_angle.get(),
                cameraFocalHeight=self.camera_focal_height.get(),
                cameraFocalLength=self.camera_focal_length.get(),
                projectionPlaneDistanceFromCenter=self.projection_plane_distance.get(),
                imageDimensions=(self.image_width.get(), self.image_height.get())
            )

            # Process videos
            status_var.set("Processing and stitching videos...")
            progress_window.update()

            # Pass both the filename and the output directory to the VideoStitcher
            output_dir = self.output_folder.get()
            video_stitcher.outputStitchedVideo(self.output_filename.get(), output_dir)

            # Close progress window
            progress_window.destroy()

            output_path = os.path.join(output_dir, self.output_filename.get() + ".mp4")
            messagebox.showinfo("Success", f"Video processing complete. Output saved to {output_path}")

        except Exception as e:
            # Close progress window
            progress_window.destroy()
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")

def main():
    root = tk.Tk()
    app = VideoStitcherUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
