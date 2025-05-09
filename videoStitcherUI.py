import os
import math
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import videoStitcher

class VideoStitcherUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Stitcher")
        self.root.geometry("800x600")

        # Video file paths
        self.left_video_path = tk.StringVar()
        self.middle_video_path = tk.StringVar()
        self.right_video_path = tk.StringVar()
        self.output_folder = tk.StringVar(value="Outputs")
        self.output_filename = tk.StringVar(value="stitched_video")

        # Camera settings
        self.left_angle = tk.IntVar(value=-30)
        self.right_angle = tk.IntVar(value=30)
        self.camera_focal_height = tk.DoubleVar(value=1.0)
        self.camera_focal_length = tk.DoubleVar(value=math.sqrt(3))
        self.projection_plane_distance = tk.DoubleVar(value=10)
        self.image_width = tk.IntVar(value=3024)
        self.image_height = tk.IntVar(value=4072)

        self._create_widgets()

    def _create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Define help texts for camera settings
        self.help_texts = {
            "left_angle": "The angle (in degrees) of the left camera relative to the center camera. Negative values rotate the camera to the left.",
            "right_angle": "The angle (in degrees) of the right camera relative to the center camera. Positive values rotate the camera to the right.",
            "focal_height": "The height of the camera's focal point above the ground plane. Higher values increase the perspective effect.",
            "focal_length": "The distance from the camera to the focal plane. Affects the field of view and perspective distortion.",
            "projection_plane": "The distance from the center camera to the projection plane. Affects how the images are stitched together.",
            "image_dimensions": "The width and height (in pixels) of the output image. Higher values result in higher resolution but require more processing time."
        }

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Video File Selection", padding="10")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        # Left video
        ttk.Label(file_frame, text="Left Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.left_video_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=lambda: self._browse_file(self.left_video_path)).grid(row=0, column=2, padx=5, pady=5)

        # Middle video
        ttk.Label(file_frame, text="Middle Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.middle_video_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=lambda: self._browse_file(self.middle_video_path)).grid(row=1, column=2, padx=5, pady=5)

        # Right video
        ttk.Label(file_frame, text="Right Video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.right_video_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=lambda: self._browse_file(self.right_video_path)).grid(row=2, column=2, padx=5, pady=5)

        # Camera settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Camera Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        # Left angle
        ttk.Label(settings_frame, text="Left Camera Angle (degrees):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(settings_frame, from_=-90, to=90, textvariable=self.left_angle, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("left_angle")).grid(row=0, column=2, padx=5, pady=5)

        # Right angle
        ttk.Label(settings_frame, text="Right Camera Angle (degrees):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(settings_frame, from_=-90, to=90, textvariable=self.right_angle, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("right_angle")).grid(row=1, column=2, padx=5, pady=5)

        # Focal height
        ttk.Label(settings_frame, text="Camera Focal Height:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(settings_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.camera_focal_height, width=10).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("focal_height")).grid(row=2, column=2, padx=5, pady=5)

        # Focal length
        ttk.Label(settings_frame, text="Camera Focal Length:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(settings_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.camera_focal_length, width=10).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("focal_length")).grid(row=3, column=2, padx=5, pady=5)

        # Projection plane distance
        ttk.Label(settings_frame, text="Projection Plane Distance:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(settings_frame, from_=1, to=100, increment=1, textvariable=self.projection_plane_distance, width=10).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="?", width=2, command=lambda: self._show_help("projection_plane")).grid(row=4, column=2, padx=5, pady=5)

        # Image dimensions
        dimensions_frame = ttk.Frame(settings_frame)
        dimensions_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(dimensions_frame, text="Image Dimensions:").pack(side=tk.LEFT)
        ttk.Label(dimensions_frame, text="Width:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(dimensions_frame, from_=100, to=10000, textvariable=self.image_width, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(dimensions_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(dimensions_frame, from_=100, to=10000, textvariable=self.image_height, width=6).pack(side=tk.LEFT, padx=5)

        # Add help button for image dimensions
        help_button_frame = ttk.Frame(settings_frame)
        help_button_frame.grid(row=5, column=2, sticky=tk.W, pady=5)
        ttk.Button(help_button_frame, text="?", width=2, command=lambda: self._show_help("image_dimensions")).pack(side=tk.LEFT)

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

    def _show_help(self, setting_key):
        """Display help information for the specified setting."""
        help_text = self.help_texts.get(setting_key, "No help available for this setting.")
        messagebox.showinfo(f"Help: {setting_key.replace('_', ' ').title()}", help_text)

    def _process_videos(self):
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

        try:
            # Create VideoStitcher instance
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
            # Pass both the filename and the output directory to the VideoStitcher
            output_dir = self.output_folder.get()
            video_stitcher.outputStitchedVideo(self.output_filename.get(), output_dir)

            output_path = os.path.join(output_dir, self.output_filename.get() + ".mp4")
            messagebox.showinfo("Success", f"Video processing complete. Output saved to {output_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")

def main():
    root = tk.Tk()
    app = VideoStitcherUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
