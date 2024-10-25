import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
import numpy as np
import threading
import time
import mediapipe as mp
import sqlite3
import logging
from PIL import Image, ImageTk
import urllib.request
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import os

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
class LoadingOverlay:
    def __init__(self, master, message="Processing..."):
        self.master = master
        self.message = message
        self.overlay = None

    def show(self):
        self.overlay = tk.Toplevel(self.master)
        self.overlay.transient(self.master)
        self.overlay.grab_set()
        self.overlay.geometry(f"{self.master.winfo_width()}x{self.master.winfo_height()}+{self.master.winfo_rootx()}+{self.master.winfo_rooty()}")
        self.overlay.overrideredirect(True)
        self.overlay.config(bg='gray')
        self.overlay.attributes('-alpha', 0.5)

        label = ttk.Label(self.overlay, text=self.message, font=("Helvetica", 16, "bold"), background='gray')
        label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.master.update_idletasks()

    def hide(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None



class EnhancedKnuckleRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced Knuckle Recognition System")
        master.geometry("1400x900")
        master.resizable(False, False)

        # Initialize status variable first
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome to the Enhanced Knuckle Recognition System")

        # Initialize all other variables
        self.knuckle_regions = None
        self.knuckle_coords = None
        self.running = False
        self.descriptors_db = {}
        self.similarity_threshold = tk.DoubleVar(value=70.0)
        self.motion_threshold = 5000
        self.cap = None
        self.current_hand_landmarks = None
        self.consecutive_matches = 0
        self.match_threshold = 3
        self.last_matched_user = None
        self.uploaded_image = None
        self.processing_image = False

        # Initialize feature extraction parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize feature extractors
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=7
        )
        
        self.sift = cv2.SIFT_create(
            nfeatures=2000,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )

        # Initialize database
        self.init_database()
        self.load_descriptors()

        # Set up GUI after initializing all variables
        self.setup_gui()

    def handle_successful_enrollment(self, user_id, successful_saves):
        """Handle successful enrollment"""
        messagebox.showinfo("Success", 
                            f"Successfully enrolled user '{user_id}' with {successful_saves} features")
        self.status_var.set(f"Enrolled user '{user_id}'")
        logging.info(f"Enrolled user '{user_id}' with {successful_saves} features")
        self.load_descriptors()

    def setup_gui(self):
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Custom styles with enhanced appearance
        self.style.configure('TButton', 
                           font=('Helvetica', 10, 'bold'),
                           padding=5)
        self.style.configure('TLabel', 
                           font=('Helvetica', 10))
        self.style.configure('Header.TLabel', 
                           font=('Helvetica', 14, 'bold'),
                           padding=10)

        # Main frames
        self.top_frame = ttk.Frame(self.master)
        self.top_frame.pack(side=tk.TOP, pady=10)

        self.middle_frame = ttk.Frame(self.master)
        self.middle_frame.pack(expand=True, fill=tk.BOTH)

        self.bottom_frame = ttk.Frame(self.master)
        self.bottom_frame.pack(side=tk.BOTTOM, pady=10)

        # Video panel with enhanced size
        self.video_frame = ttk.Frame(self.middle_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10)
        
        self.video_panel = ttk.Label(self.video_frame)
        self.video_panel.pack()

        # Knuckle panels with labels
        self.knuckle_frame = ttk.Frame(self.middle_frame)
        self.knuckle_frame.pack(side=tk.LEFT, padx=10)
        
        self.knuckle_panels = {}
        self.knuckle_names = ['knuckle_1', 'knuckle_2', 'knuckle_3', 'knuckle_4']
        
        for name in self.knuckle_names:
            panel_frame = ttk.Frame(self.knuckle_frame)
            panel_frame.pack(pady=5)
            
            label = ttk.Label(panel_frame, text=f"Knuckle {name[-1]}")
            label.pack()
            
            panel = ttk.Label(panel_frame)
            panel.pack()
            self.knuckle_panels[name] = panel

        # Enhanced Camera Settings Frame
        self.camera_frame = ttk.LabelFrame(self.top_frame, text="Camera Settings")
        self.camera_frame.grid(row=2, column=0, columnspan=4, pady=5, padx=5)
        
        self.url_label = ttk.Label(self.camera_frame, text="IP Camera URL:")
        self.url_label.pack(side=tk.LEFT, padx=5)
        
        self.url_entry = ttk.Entry(self.camera_frame, width=50)
        self.url_entry.pack(side=tk.LEFT, padx=5)
        self.url_entry.insert(0, "http://192.168.1.100:8080/video")

        # Enhanced Control Buttons Frame
        self.button_frame = ttk.LabelFrame(self.top_frame, text="Controls")
        self.button_frame.grid(row=0, column=0, columnspan=4, pady=5)

        # Camera Control Buttons
        self.camera_controls = ttk.Frame(self.button_frame)
        self.camera_controls.pack(pady=5)

        self.start_button = ttk.Button(
            self.camera_controls, text="Start Camera", 
            command=self.start_webcam
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            self.camera_controls, text="Stop Camera",
            command=self.stop_webcam,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Image Upload Controls
        self.upload_frame = ttk.Frame(self.button_frame)
        self.upload_frame.pack(pady=5)

        self.upload_button = ttk.Button(
            self.upload_frame,
            text="Upload Image",
            command=self.upload_image
        )
        self.upload_button.pack(side=tk.LEFT, padx=5)

        # Enrollment and Matching Controls
        self.action_frame = ttk.Frame(self.button_frame)
        self.action_frame.pack(pady=5)

        self.capture_button = ttk.Button(
            self.action_frame, text="Enroll Hand",
            command=self.capture_knuckle,
            state=tk.DISABLED
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)

        self.enroll_image_button = ttk.Button(
            self.action_frame,
            text="Enroll from Image",
            command=self.enroll_from_image,
            state=tk.DISABLED
        )
        self.enroll_image_button.pack(side=tk.LEFT, padx=5)

        self.match_button = ttk.Button(
            self.action_frame, text="Match Hand",
            command=self.match_knuckle,
            state=tk.DISABLED
        )
        self.match_button.pack(side=tk.LEFT, padx=5)

        # Enhanced threshold controls
        self.threshold_frame = ttk.LabelFrame(self.top_frame, text="Matching Settings")
        self.threshold_frame.grid(row=1, column=0, columnspan=4, pady=5, padx=5)

        self.threshold_label = ttk.Label(self.threshold_frame, text="Similarity Threshold:")
        self.threshold_label.pack(side=tk.LEFT, padx=5)

        self.threshold_slider = ttk.Scale(
            self.threshold_frame,
            from_=0.0,
            to=100.0,
            orient='horizontal',
            variable=self.similarity_threshold,
            command=self.update_threshold_label,
            length=200
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        self.threshold_value_label = ttk.Label(
            self.threshold_frame,
            text=f"{self.similarity_threshold.get():.2f}%"
        )
        self.threshold_value_label.pack(side=tk.LEFT, padx=5)

        # Results section
        self.result_frame = ttk.LabelFrame(self.bottom_frame, text="Results")
        self.result_frame.pack(fill=tk.X, padx=10)

        self.result_label = ttk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 14)
        )
        self.result_label.pack(pady=5)

        # Enhanced progress bar
        self.progress_bar = ttk.Progressbar(
            self.result_frame,
            orient='horizontal',
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)

        # Status bar with enhanced visibility
        self.status_bar = ttk.Label(
            self.master,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_threshold_label(self, event):
        self.threshold_value_label.config(text=f"{self.similarity_threshold.get():.2f}%")

    def upload_image(self):
        """Handle image upload and processing"""
        try:
            # Open file dialog for image selection
            file_path = filedialog.askopenfilename(
                title="Select Hand Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return

            # Load and process image
            self.uploaded_image = cv2.imread(file_path)
            if self.uploaded_image is None:
                raise Exception("Failed to load image")

            # Resize image while maintaining aspect ratio
            height, width = self.uploaded_image.shape[:2]
            max_dim = 800
            scale = min(max_dim/width, max_dim/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.uploaded_image = cv2.resize(self.uploaded_image, (new_width, new_height))

            # Stop webcam if running
            if self.running:
                self.stop_webcam()

            self.processing_image = True
            # Process uploaded image
            self.process_uploaded_image()

        except Exception as e:
            logging.exception(f"Error uploading image: {e}")
            messagebox.showerror("Error", f"Failed to upload image: {str(e)}")


    def capture_knuckle(self):
        """Enhanced knuckle capture with comprehensive error handling"""
        try:
            if not self.verify_capture_readiness():
                return

            user_id = self.get_user_id()
            if not user_id:
                return

            self.status_var.set(f"Capturing samples for '{user_id}'")
            samples = self.collect_samples()
            
            if not samples:
                return

            if self.verify_samples(samples):
                self.save_samples(user_id, samples)
            else:
                messagebox.showerror("Error", "Samples failed quality verification")

        except Exception as e:
            logging.exception(f"Critical error during enrollment: {e}")
            messagebox.showerror("Error", f"Enrollment failed: {str(e)}")

    def verify_capture_readiness(self):
        """Check if system is ready for capture"""
        if self.knuckle_regions is None:
            messagebox.showerror("Error", 
                "No knuckle regions detected. Please ensure your hand is clearly visible.")
            return False

        if not self.current_hand_landmarks:
            messagebox.showerror("Error", 
                "Hand landmarks not detected. Please adjust your hand position.")
            return False

        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not accessible")
            return False

        return True

    def get_user_id(self):
        """Get and validate user ID"""
        user_id = simpledialog.askstring("Input", "Enter your name or ID:", parent=self.master)
        if not user_id:
            messagebox.showwarning("Cancelled", "Enrollment cancelled.")
            return None
            
        if not self.validate_user_id(user_id):
            messagebox.showerror("Error", "Invalid user ID format")
            return None
            
        return user_id

    def validate_user_id(self, user_id):
        """Validate user ID format"""
        # Basic validation rules
        if not user_id or len(user_id) < 3 or len(user_id) > 50:
            return False
        if not user_id.replace(" ", "").isalnum():
            return False
        return True

    def collect_samples(self):
        """Collect multiple samples for enrollment"""
        samples = []
        max_samples = 5
        captured = 0
        capture_timeout = time.time() + 10  # 10 second timeout

        while captured < max_samples and self.running:
            if time.time() > capture_timeout:
                messagebox.showerror("Error", "Capture timeout. Please try again.")
                return None

            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Failed to capture frame")

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    knuckle_regions, _ = self.extract_knuckle_regions(frame, hand_landmarks)

                    if knuckle_regions and self.verify_image_quality(knuckle_regions):
                        samples.append(knuckle_regions)
                        captured += 1
                        self.status_var.set(f"Captured sample {captured}/{max_samples}")
                        time.sleep(0.5)  # Delay between captures

            except Exception as e:
                logging.exception(f"Error during sample collection: {e}")
                continue

        if len(samples) < 2:
            messagebox.showerror("Error", "Could not capture enough samples. Please try again.")
            return None

        return samples

    def verify_samples(self, samples):
        """Verify quality and motion in samples"""
        try:
            # Check for motion between samples
            total_diff = 0
            for i in range(len(samples)-1):
                sample_diff = 0
                for knuckle_name in self.knuckle_names:
                    if (knuckle_name in samples[i] and 
                        knuckle_name in samples[i+1]):
                        diff = cv2.absdiff(samples[i][knuckle_name], 
                                         samples[i+1][knuckle_name])
                        sample_diff += np.sum(diff)
                total_diff += sample_diff

            avg_motion = total_diff / (len(samples) - 1)
            if avg_motion < self.motion_threshold:
                messagebox.showerror("Error", 
                    "Insufficient hand movement detected. Please move your hand naturally.")
                return False

            # Verify quality of all samples
            for sample in samples:
                if not self.verify_image_quality(sample):
                    return False

            return True

        except Exception as e:
            logging.exception(f"Error in sample verification: {e}")
            return False

    def save_samples(self, user_id, samples):
        """Save verified samples to database"""
        try:
            successful_saves = 0
            for sample_idx, sample in enumerate(samples):
                for knuckle_name, knuckle_image in sample.items():
                    descriptors = self.get_enhanced_descriptors(knuckle_image)
                    if descriptors is not None and len(descriptors) > 0:
                        if self.save_descriptors_to_db(user_id, knuckle_name, descriptors):
                            successful_saves += 1
                    else:
                        logging.warning(f"No features extracted from {knuckle_name} in sample {sample_idx + 1}")

            if successful_saves > 0:
                messagebox.showinfo("Success", 
                    f"Successfully enrolled user '{user_id}' with {successful_saves//(len(samples)*4)} complete samples")
                self.status_var.set(f"Enrolled user '{user_id}'")
                logging.info(f"Enrolled user '{user_id}' with {successful_saves} descriptors")
                self.load_descriptors()
            else:
                raise Exception("No samples were successfully saved")

        except Exception as e:
            logging.exception(f"Error saving samples: {e}")
            messagebox.showerror("Error", f"Failed to save samples: {str(e)}")
    
    def get_enhanced_descriptors(self, image):
        """Extract features using only ORB for consistency"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Create ORB detector with optimized parameters
            orb = cv2.ORB_create(
                nfeatures=5000,        # Increased number of features
                scaleFactor=1.1,       # More precise scale factor
                nlevels=12,           # More levels for better detection
                edgeThreshold=10,      # Lower threshold for better edge detection
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=21,          # Smaller patch size
                fastThreshold=5        # Lower threshold for more features
            )
            
            # Detect and compute features
            keypoints, descriptors = orb.detectAndCompute(processed_image, None)
            
            if keypoints is None or len(keypoints) == 0:
                logging.warning("No keypoints detected")
                return None
            
            if descriptors is None:
                logging.warning("No descriptors computed")
                return None
                
            logging.info(f"Extracted {len(keypoints)} keypoints with descriptors shape: {descriptors.shape}")
            return descriptors
            
        except Exception as e:
            logging.exception(f"Error in feature extraction: {e}")
            return None

    def extract_knuckle_regions(self, image, hand_landmarks):
        """Extract and process knuckle regions with enhanced error handling"""
        try:
            knuckle_regions = {}
            knuckle_coords = {}
            padding = 30  # Base padding size
            
            # Define knuckle landmark pairs
            knuckle_pairs = {
                'knuckle_1': (5, 6),    # Index finger
                'knuckle_2': (9, 10),   # Middle finger
                'knuckle_3': (13, 14),  # Ring finger
                'knuckle_4': (17, 18)   # Little finger
            }

            for knuckle_name, (start_idx, end_idx) in knuckle_pairs.items():
                try:
                    # Get landmark coordinates
                    if not (hasattr(hand_landmarks.landmark[start_idx], 'x') and 
                           hasattr(hand_landmarks.landmark[end_idx], 'x')):
                        continue

                    x1 = int(hand_landmarks.landmark[start_idx].x * image.shape[1])
                    y1 = int(hand_landmarks.landmark[start_idx].y * image.shape[0])
                    x2 = int(hand_landmarks.landmark[end_idx].x * image.shape[1])
                    y2 = int(hand_landmarks.landmark[end_idx].y * image.shape[0])

                    # Calculate dynamic padding based on knuckle size
                    knuckle_size = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                    dynamic_padding = max(padding, int(knuckle_size * 0.5))

                    # Compute bounding box with dynamic padding
                    x_min = max(0, min(x1, x2) - dynamic_padding)
                    x_max = min(image.shape[1], max(x1, x2) + dynamic_padding)
                    y_min = max(0, min(y1, y2) - dynamic_padding)
                    y_max = min(image.shape[0], max(y1, y2) + dynamic_padding)

                    # Validate region size
                    if x_max - x_min <= 0 or y_max - y_min <= 0:
                        logging.warning(f"Invalid region size for {knuckle_name}")
                        continue

                    # Extract region
                    knuckle_region = image[y_min:y_max, x_min:x_max].copy()
                    if knuckle_region.size == 0:
                        continue

                    # Align and enhance region
                    aligned_region = self.align_knuckle_image(
                        knuckle_region,
                        x1 - x_min,
                        y1 - y_min,
                        x2 - x_min,
                        y2 - y_min
                    )

                    # Resize to standard size
                    resized_region = cv2.resize(aligned_region, (224, 224))

                    # Store regions and coordinates
                    knuckle_regions[knuckle_name] = resized_region
                    knuckle_coords[knuckle_name] = (x_min, y_min, x_max, y_max)

                    # Draw visualization
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, knuckle_name, (x_min, y_min-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    logging.exception(f"Error processing {knuckle_name}: {e}")
                    continue

            if not knuckle_regions:
                logging.warning("No valid knuckle regions extracted")
                return None, None

            return knuckle_regions, knuckle_coords

        except Exception as e:
            logging.exception(f"Error in knuckle extraction: {e}")
            return None, None

    def align_knuckle_image(self, image, x1, y1, x2, y2):
        """Align knuckle image with enhanced error handling"""
        try:
            # Calculate rotation angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Get image dimensions
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            
            # Calculate new dimensions after rotation
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            
            # Adjust rotation matrix for new dimensions
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Perform rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            
            # Apply perspective correction for significant angles
            if abs(angle) > 30:
                src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                dst_pts = np.float32([[10, 0], [width-10, 0], [0, height], [width, height]])
                perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                rotated = cv2.warpPerspective(rotated, perspective_matrix, (width, height))

            return rotated

        except Exception as e:
            logging.exception(f"Error in knuckle alignment: {e}")
            return image  # Return original image if alignment fails

    def process_uploaded_image(self):
        """Process the uploaded image and detect hand landmarks"""
        try:
            if self.uploaded_image is None:
                raise Exception("No image uploaded")

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(self.uploaded_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                raise Exception("No hand detected in image")

            # Get hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            self.current_hand_landmarks = hand_landmarks
            
            # Extract knuckle regions
            self.knuckle_regions, self.knuckle_coords = self.extract_knuckle_regions(
                self.uploaded_image.copy(), 
                hand_landmarks
            )

            if not self.knuckle_regions:
                raise Exception("Failed to extract knuckle regions")

            # Verify image quality
            if not self.verify_image_quality(self.knuckle_regions):
                raise Exception("Image quality too low for processing")

            # Display processed image
            self.display_processed_image()
            
            # Enable enrollment button
            self.enroll_image_button.config(state=tk.NORMAL)
            self.match_button.config(state=tk.NORMAL)
            self.status_var.set("Image processed successfully. Ready for enrollment.")

        except Exception as e:
            logging.exception(f"Error processing image: {e}")
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.enroll_image_button.config(state=tk.DISABLED)
            self.match_button.config(state=tk.DISABLED)

    def display_processed_image(self):
        """Display the processed image with detected knuckles"""
        try:
            # Create a copy of the image for display
            display_image = self.uploaded_image.copy() if self.processing_image else self.current_frame

            # Draw hand landmarks
            if self.current_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    display_image,
                    self.current_hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Draw knuckle regions with labels
            if self.knuckle_coords:
                for name, (x_min, y_min, x_max, y_max) in self.knuckle_coords.items():
                    cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(display_image, name, (x_min, y_min-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Convert to RGB for display
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            # Create PhotoImage
            image_pil = Image.fromarray(display_image_rgb)
            image_pil = image_pil.resize((900, 600), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update video panel
            self.video_panel.config(image=image_tk)
            self.video_panel.image = image_tk

            # Display knuckle regions
            if self.knuckle_regions:
                for name, region in self.knuckle_regions.items():
                    # Enhanced preprocessing for display
                    processed_region = self.enhance_knuckle_display(region)
                    knuckle_pil = Image.fromarray(processed_region)
                    knuckle_pil = knuckle_pil.resize((150, 150))
                    knuckle_tk = ImageTk.PhotoImage(knuckle_pil)
                    self.knuckle_panels[name].config(image=knuckle_tk)
                    self.knuckle_panels[name].image = knuckle_tk

        except Exception as e:
            logging.exception(f"Error displaying processed image: {e}")
            messagebox.showerror("Error", f"Failed to display processed image: {str(e)}")

    def enhance_knuckle_display(self, region):
        """Enhance knuckle region for better visualization"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Apply adaptive histogram equalization
            equ = cv2.equalizeHist(denoised)
            
            # Create color map for better visualization
            colored = cv2.applyColorMap(equ, cv2.COLORMAP_BONE)
            
            # Enhance edges
            edges = cv2.Canny(denoised, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Combine original and edge detection
            result = cv2.addWeighted(colored, 0.7, edges_colored, 0.3, 0)
            
            return result
            
        except Exception as e:
            logging.exception(f"Error enhancing knuckle display: {e}")
            return cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    def extract_enhanced_features(self, image):
        """Extract multiple types of features without concatenation"""
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            features_dict = {}
            
            # Extract ORB features
            orb_kp, orb_desc = self.orb.detectAndCompute(processed, None)
            if orb_desc is not None:
                features_dict['orb'] = orb_desc
            
            # Extract SIFT features
            sift_kp, sift_desc = self.sift.detectAndCompute(processed, None)
            if sift_desc is not None:
                features_dict['sift'] = sift_desc
            
            # Extract corner features (if applicable)
            corners = cv2.goodFeaturesToTrack(processed, **self.feature_params)
            if corners is not None:
                corners = corners.reshape(-1, 2)
                features_dict['corners'] = corners
            
            if not features_dict:
                return None  # No features extracted
            
            return features_dict
            
        except Exception as e:
            logging.exception(f"Error extracting enhanced features: {e}")
            return None




    def verify_image_quality(self, regions):
        """Verify image quality with adjusted thresholds and feedback"""
        try:
            for name, region in regions.items():
                if region is None or region.size == 0:
                    logging.warning(f"Invalid region for {name}")
                    messagebox.showwarning("Image Quality Issue", f"Invalid region detected for {name}. Please adjust your hand position.")
                    return False

                # Convert to grayscale if needed
                if len(region.shape) == 3:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    gray = region.copy()

                # Calculate quality metrics
                min_val, max_val, _, _ = cv2.minMaxLoc(gray)
                contrast = max_val - min_val
                if contrast < 20:
                    logging.warning(f"Low contrast in {name}: {contrast}")
                    messagebox.showwarning("Image Quality Issue", f"Low contrast detected in {name}. Please improve lighting.")
                    return False

                # Check brightness
                mean_val = np.mean(gray)
                if mean_val < 15 or mean_val > 240:
                    logging.warning(f"Poor brightness in {name}: {mean_val}")
                    messagebox.showwarning("Image Quality Issue", f"Poor brightness detected in {name}. Please adjust lighting.")
                    return False

                # Check blur level
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 30:
                    logging.warning(f"Blurry image detected in {name}: {laplacian_var}")
                    messagebox.showwarning("Image Quality Issue", f"Blurry image detected in {name}. Please keep your hand steady.")
                    return False

            return True

        except Exception as e:
            logging.exception(f"Error in image quality verification: {e}", exc_info=True)
            messagebox.showerror("Error", f"Image quality verification failed: {str(e)}")
            return False


    def enhance_image_quality(self, image):
        """Enhance image quality before processing"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, 
                                               h=10,  # Reduced from default
                                               templateWindowSize=7,
                                               searchWindowSize=21)

            # Sharpen
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Normalize
            normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

            return normalized

        except Exception as e:
            logging.exception(f"Error enhancing image quality: {e}")
            return image

    def preprocess_for_quality_check(self, image):
        """Preprocess image before quality verification"""
        try:
            # Basic preprocessing
            processed = self.enhance_image_quality(image)
            
            # Edge enhancement
            edges = cv2.Canny(processed, 50, 150)
            processed = cv2.addWeighted(processed, 0.7, edges, 0.3, 0)

            return processed

        except Exception as e:
            logging.exception(f"Error in preprocessing: {e}")
            return image

    def estimate_noise_level(self, image):
        """Estimate image noise level using multiple methods"""
        try:
            # Method 1: Using local standard deviation
            kernel_size = 5
            local_mean = cv2.blur(image, (kernel_size, kernel_size))
            local_std = np.zeros_like(image, dtype=np.float32)
            cv2.blur((image - local_mean) ** 2, (kernel_size, kernel_size), local_std)
            noise_std = np.mean(np.sqrt(local_std))

            # Method 2: Using Laplacian
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_lap = np.mean(np.abs(laplacian))

            # Combine both metrics
            noise_level = (noise_std + noise_lap) / 2
            return noise_level

        except Exception as e:
            logging.exception(f"Error estimating noise level: {e}")
            return 100  # Return high noise level on error
    def calculate_local_contrast(self, image, window_size=15):
        """Calculate local contrast using sliding window"""
        try:
            height, width = image.shape
            pad_size = window_size // 2
            padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                      cv2.BORDER_REFLECT)
            contrast_map = np.zeros((height, width))

            for i in range(height):
                for j in range(width):
                    window = padded[i:i+window_size, j:j+window_size]
                    contrast_map[i, j] = (np.max(window) - np.min(window)) / (np.max(window) + np.min(window) + 1e-6)

            return np.mean(contrast_map)

        except Exception as e:
            logging.exception(f"Error calculating local contrast: {e}")
            return 0
    



    def enroll_from_image(self):
        """Enroll user using the uploaded image"""
        try:
            if not self.verify_enrollment_readiness():
                return

            user_id = self.get_user_id()
            if not user_id:
                return

            self.status_var.set(f"Processing enrollment for '{user_id}'...")

            # Extract and verify features
            features_data = {}
            for knuckle_name, region in self.knuckle_regions.items():
                features_dict = self.extract_enhanced_features(region)
                if features_dict is not None and len(features_dict) > 0:
                    features_data[knuckle_name] = features_dict
                else:
                    raise Exception(f"No features extracted from {knuckle_name}")

            # Save features to database
            successful_saves = self.save_enrollment_data(user_id, features_data)

            if successful_saves > 0:
                self.handle_successful_enrollment(user_id, successful_saves)
            else:
                raise Exception("No features were successfully saved")

        except Exception as e:
            self.handle_enrollment_error(e)


    def verify_enrollment_readiness(self):
        """Check if system is ready for enrollment"""
        if self.knuckle_regions is None:
            messagebox.showerror("Error", "No knuckle regions detected")
            return False

        if not self.verify_image_quality(self.knuckle_regions):
            messagebox.showerror("Error", "Image quality too low for enrollment")
            return False

        return True

    def get_user_id(self):
        """Get user ID with validation"""
        user_id = simpledialog.askstring("Input", "Enter your name or ID:", parent=self.master)
        if not user_id:
            return None

        # Validate user ID
        if not self.validate_user_id(user_id):
            messagebox.showerror("Error", "Invalid user ID format")
            return None

        return user_id

    def validate_user_id(self, user_id):
        """Validate user ID format"""
        # Add your validation rules here
        return len(user_id) >= 3 and len(user_id) <= 50 and user_id.isprintable()

    def save_enrollment_data(self, user_id, features_data):
        """Save enrollment data to database"""
        successful_saves = 0
        with self.conn:
            # Insert or update user
            self.cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, enrollment_date) 
                VALUES (?, CURRENT_TIMESTAMP)
            ''', (user_id,))

            # Save features for each knuckle
            for knuckle_name, features_dict in features_data.items():
                quality_score = self.calculate_feature_quality(features_dict)
                
                if self.save_descriptors_to_db(user_id, knuckle_name, features_dict):
                    successful_saves += 1
        return successful_saves



    def preprocess_image(self, image):
        """Enhanced preprocessing for better feature detection"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Denoise with reduced parameters
            denoised = cv2.fastNlMeansDenoising(enhanced, h=5)  # Reduced denoising strength

            # Apply adaptive histogram equalization
            equ = cv2.equalizeHist(denoised)

            # Sharpen the image
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(equ, -1, kernel)

            # Normalize
            normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

            return normalized

        except Exception as e:
            logging.exception(f"Error in preprocessing: {e}")
            return image

    def match_descriptors(self, current_desc_dict, stored_desc_dict):
        """Match descriptors of the same type separately and aggregate results"""
        total_matches = 0
        for feature_type in current_desc_dict:
            if feature_type in stored_desc_dict:
                current_desc = current_desc_dict[feature_type]
                stored_desc_list = stored_desc_dict[feature_type]
                for stored_desc in stored_desc_list:
                    matches = self.match_single_descriptor(current_desc, stored_desc, feature_type)
                    total_matches += matches
        return total_matches

    def match_single_descriptor(self, desc1, desc2, feature_type):
        """Match descriptors based on their feature type."""
        try:
            if feature_type == 'orb':
                # Use Hamming distance for binary descriptors
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(desc1, desc2, k=2)
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                return len(good_matches)
            elif feature_type == 'sift':
                # Use Euclidean distance for float descriptors
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(desc1, desc2, k=2)
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                return len(good_matches)
            elif feature_type == 'corners':
                # For corner features, you might need a different approach
                # This is a placeholder
                return 0
            else:
                logging.warning(f"Unknown feature type: {feature_type}")
                return 0
        except Exception as e:
            logging.exception(f"Error matching descriptors: {e}")
            return 0
        

    def save_descriptors_to_db(self, user_id, knuckle_name, descriptors_dict):
        """Save descriptors to database separately."""
        try:
            for feature_type, descriptors in descriptors_dict.items():
                if descriptors is None or len(descriptors) == 0:
                    logging.exception(f"Empty descriptors for {knuckle_name} with feature type {feature_type}")
                    continue

                # Calculate quality score for each feature type
                quality_score = self.calculate_feature_quality({feature_type: descriptors})

                # Convert descriptors to bytes
                if descriptors.dtype != np.uint8:
                    # Convert to bytes appropriately for float descriptors
                    descriptors_bytes = descriptors.astype(np.float32).tobytes()
                else:
                    descriptors_bytes = descriptors.tobytes()

                # Save descriptors
                self.cursor.execute('''
                    INSERT INTO user_descriptors 
                    (user_id, knuckle_name, descriptors, feature_type, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, knuckle_name, descriptors_bytes, feature_type, quality_score))

            # Do not commit here; let the caller handle it
            return True
        except Exception as e:
            logging.exception(f"Error saving descriptors to database: {e}")
            # Re-raise the exception to be handled by the caller
            raise e


    def calculate_feature_quality(self, features_dict):
        """Calculate quality score for features"""
        if features_dict is None:
            return 0

        total_features = 0
        total_variance = 0
        total_entropy = 0

        for feature_type, features in features_dict.items():
            num_features = len(features)
            feature_variance = np.var(features)
            feature_entropy = np.sum(np.abs(np.fft.fft2(features)))

            total_features += num_features
            total_variance += feature_variance
            total_entropy += feature_entropy

        # Combine metrics into a single score
        quality_score = (total_features * 0.4 +
                        total_variance * 0.3 +
                        total_entropy * 0.3)

        return quality_score

    def load_descriptors(self):
        """Load user descriptors from database."""
        try:
            self.cursor.execute('SELECT user_id FROM users')
            users = [row[0] for row in self.cursor.fetchall()]
            self.descriptors_db = {}
            
            for user_id in users:
                self.cursor.execute('''
                    SELECT knuckle_name, descriptors, feature_type, quality_score 
                    FROM user_descriptors 
                    WHERE user_id = ? 
                    ORDER BY quality_score DESC
                ''', (user_id,))
                
                user_knuckle_descriptors = {}
                for row in self.cursor.fetchall():
                    knuckle_name = row[0]
                    descriptors_blob = row[1]
                    feature_type = row[2]
                    
                    # Determine descriptor size
                    if feature_type == 'orb':
                        descriptor_size = 32
                        dtype = np.uint8
                    elif feature_type == 'sift':
                        descriptor_size = 128
                        dtype = np.float32
                    else:
                        continue  # Skip unknown feature types
                    
                    descriptors = np.frombuffer(descriptors_blob, dtype=dtype).reshape(-1, descriptor_size)
                    
                    # Organize descriptors by knuckle and feature type
                    if knuckle_name not in user_knuckle_descriptors:
                        user_knuckle_descriptors[knuckle_name] = {}
                    if feature_type not in user_knuckle_descriptors[knuckle_name]:
                        user_knuckle_descriptors[knuckle_name][feature_type] = []
                    user_knuckle_descriptors[knuckle_name][feature_type].append(descriptors)
                
                if user_knuckle_descriptors:
                    self.descriptors_db[user_id] = user_knuckle_descriptors
            
            num_users = len(self.descriptors_db)
            logging.info(f"Loaded descriptors for {num_users} users")
            self.status_var.set(f"Loaded {num_users} user profiles")
            
        except Exception as e:
            logging.exception(f"Error loading descriptors: {e}")
            messagebox.showerror("Error", f"Failed to load descriptors: {str(e)}")
            self.descriptors_db = {}
  # Initialize empty if loading fails
   
    def start_webcam(self):
        """Start webcam capture with enhanced error handling"""
        if self.running:
            return
            
        camera_url = self.url_entry.get().strip()
        
        if not camera_url:
            messagebox.showerror("Error", "Please enter IP camera URL")
            return
            
        try:
            # Initialize camera capture
            self.cap = cv2.VideoCapture(camera_url)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test connection
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("Cannot read from IP camera")
                
            if not self.cap.isOpened():
                raise Exception("Could not open camera stream")
                
        except Exception as e:
            error_msg = (f"Could not connect to IP camera: {str(e)}\n\n"
                        f"Make sure:\n"
                        f"1. Your phone and computer are on the same WiFi network\n"
                        f"2. The IP camera app is running on your phone\n"
                        f"3. The URL is correct")
            messagebox.showerror("Error", error_msg)
            self.cap = None
            logging.exception(f"Camera connection error: {e}")
            return
            
        # Start capture
        self.running = True
        self.update_button_states(capturing=True)
        
        # Start frame update thread
        threading.Thread(target=self.update_frame, daemon=True).start()
        
        logging.info("IP camera started")
        self.status_var.set("IP camera connected and started")

    def stop_webcam(self):
        """Stop webcam capture"""
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear displays
        self.clear_displays()
        
        # Reset buttons
        self.update_button_states(capturing=False)
        
        logging.info("Camera stopped")
        self.status_var.set("Camera stopped")

    def update_button_states(self, capturing=False):
        """Update button states based on capture status"""
        if capturing:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.capture_button.config(state=tk.NORMAL)
            self.match_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.capture_button.config(state=tk.DISABLED)
            self.match_button.config(state=tk.DISABLED)

    def clear_displays(self):
        """Clear all display panels"""
        self.video_panel.config(image='')
        for panel in self.knuckle_panels.values():
            panel.config(image='')
        self.result_label.config(text='')
        self.progress_bar['value'] = 0

    def update_frame(self):
        """Update frame from camera with enhanced error handling"""
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    raise Exception("Camera connection lost")

                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Failed to read frame")

                # Process frame
                frame = cv2.flip(frame, 1)  # Mirror horizontally
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hand landmarks
                results = self.hands.process(image_rgb)
                self.process_hand_landmarks(frame, results)

                # Update video display
                self.update_video_display(frame)

            except Exception as e:
                logging.exception(f"Frame update error: {e}")
                self.status_var.set(f"Camera error: {str(e)}")
                self.stop_webcam()
                break

            time.sleep(0.03)  # Limit frame rate

    def process_hand_landmarks(self, frame, results):
        """Process detected hand landmarks"""
        self.knuckle_regions = None
        self.current_hand_landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.current_hand_landmarks = hand_landmarks
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Extract knuckle regions
            self.knuckle_regions, self.knuckle_coords = self.extract_knuckle_regions(frame, hand_landmarks)
            
            # Update displays
            if self.knuckle_regions:
                self.result_label.config(text="Knuckles detected", foreground='green')
                self.update_knuckle_displays()
            else:
                self.result_label.config(text="Knuckles not detected", foreground='red')
                self.clear_knuckle_displays()
        else:
            self.result_label.config(text="Hand not detected", foreground='red')
            self.clear_knuckle_displays()

    def update_video_display(self, frame):
        """Update the main video display"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((900, 600))
        frame_tk = ImageTk.PhotoImage(frame_pil)
        self.video_panel.config(image=frame_tk)
        self.video_panel.image = frame_tk

    def update_knuckle_displays(self):
        """Update the knuckle region displays"""
        for name, region in self.knuckle_regions.items():
            processed_region = self.preprocess_image(region)
            display_image = cv2.cvtColor(processed_region, cv2.COLOR_GRAY2RGB)
            knuckle_pil = Image.fromarray(display_image)
            knuckle_pil = knuckle_pil.resize((150, 150))
            knuckle_tk = ImageTk.PhotoImage(knuckle_pil)
            self.knuckle_panels[name].config(image=knuckle_tk)
            self.knuckle_panels[name].image = knuckle_tk

    def match_knuckle_thread(self, loading):
        # The match_knuckle_thread method should be indented here
        try:
            match_data = self.perform_matching()
            if not match_data:
                self.handle_no_match()
            else:
                match_results, total_possible_matches = match_data
                self.process_match_results(match_results, total_possible_matches)
        except Exception as e:
            self.handle_matching_error(e)
        finally:
            # Hide loading overlay and re-enable the match button
            loading.hide()
            self.match_button.config(state=tk.NORMAL)
            self.master.update()

    def match_knuckle(self):
        """Enhanced matching with loading indicator and improved error handling"""
        try:
            if not self.verify_matching_readiness():
                return

            # Show loading overlay
            loading = LoadingOverlay(self.master, message="Matching in progress...")
            loading.show()
            self.master.update()

            # Disable the match button to prevent multiple clicks
            self.match_button.config(state=tk.DISABLED)

            # Perform matching in a separate thread
            threading.Thread(target=self.match_knuckle_thread, args=(loading,), daemon=True).start()

        except Exception as e:
            self.handle_matching_error(e)


    def clear_knuckle_displays(self):
        """Clear all knuckle displays"""
        for panel in self.knuckle_panels.values():
            panel.config(image='')
    def init_database(self):
        """Initialize database with corrected schema"""
        try:
            self.conn = sqlite3.connect('enhanced_knuckle_users.db')
            self.cursor = self.conn.cursor()
            
            # Enable foreign keys
            self.cursor.execute('PRAGMA foreign_keys = ON')
            
            # Create users table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_match_date TIMESTAMP,
                    match_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Create descriptors table with use_count
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_descriptors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    knuckle_name TEXT,
                    descriptors BLOB,
                    feature_type TEXT,
                    quality_score REAL,
                    use_count INTEGER DEFAULT 0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            # Create indices
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_descriptors_user_id 
                ON user_descriptors(user_id)
            ''')
            
            self.conn.commit()
            logging.info("Database initialized successfully")
            
        except Exception as e:
            logging.exception(f"Database initialization error: {e}")
            raise
    


    def handle_enrollment_error(self, error):
        """Handle enrollment errors"""
        logging.exception(f"Enrollment error: {error}")
        messagebox.showerror("Error", f"Enrollment failed: {str(error)}")
        self.status_var.set("Enrollment failed")



    def verify_matching_readiness(self):
        """Verify system is ready for matching"""
        if self.knuckle_regions is None:
            messagebox.showerror("Error", "No knuckle regions detected")
            return False

        if not self.descriptors_db:
            messagebox.showinfo("Match Result", "No enrolled users to match against")
            return False

        return True

    def perform_matching(self):
        """Perform the matching process"""
        current_descriptors = {}
        match_results = {}
        total_possible_matches = 0

        # Get current descriptors
        for knuckle_name, knuckle_image in self.knuckle_regions.items():
            features = self.extract_enhanced_features(knuckle_image)
            if features is not None:
                current_descriptors[knuckle_name] = features
                # Calculate total possible matches based on the number of descriptors
                for feature_type, desc in features.items():
                    if desc is not None:
                        total_possible_matches += len(desc)

        if not current_descriptors:
            raise Exception("Could not extract features from current knuckles")

        # Match against database
        for user_id, user_data in self.descriptors_db.items():
            scores = self.match_against_user(current_descriptors, user_data)
            if scores:
                match_results[user_id] = scores

        return (match_results, total_possible_matches) if match_results else None

    def match_against_user(self, current_descriptors, user_data):
        """Match current descriptors against a single user."""
        user_scores = {}
        
        for knuckle_name, current_desc_dict in current_descriptors.items():
            if knuckle_name in user_data:
                stored_desc_types = user_data[knuckle_name]
                for feature_type in current_desc_dict:
                    if feature_type in stored_desc_types:
                        current_desc = current_desc_dict[feature_type]
                        stored_desc_list = stored_desc_types[feature_type]
                        best_score = self.get_best_match_score(current_desc, stored_desc_list, feature_type)
                        if best_score > 0:
                            if knuckle_name not in user_scores:
                                user_scores[knuckle_name] = 0
                            user_scores[knuckle_name] += best_score
        
        if user_scores:
            return {
                'scores': user_scores,
                'total_score': sum(user_scores.values())
            }
        return None


    def get_best_match_score(self, current_desc, stored_desc_list, feature_type):
        """Get the best match score from multiple stored descriptors."""
        scores = []
        for stored_desc in stored_desc_list:
            score = self.match_single_descriptor(current_desc, stored_desc, feature_type)
            scores.append(score)
        return max(scores) if scores else 0


    def process_match_results(self, match_results, total_possible_matches):
        """Process and display match results"""
        best_match_user = max(match_results, key=lambda k: match_results[k]['total_score'])
        best_match_score = match_results[best_match_user]['total_score']

        # Calculate confidence
        confidence = (best_match_score / total_possible_matches) * 100
        self.progress_bar['value'] = confidence

        if confidence >= self.similarity_threshold.get():
            self.handle_positive_match(best_match_user, confidence, match_results[best_match_user])
        else:
            self.handle_no_match()

    def handle_positive_match(self, user_id, confidence, match_data):
        """Handle positive match result"""
        # Update consecutive matches
        if user_id == self.last_matched_user:
            self.consecutive_matches += 1
        else:
            self.consecutive_matches = 1
            self.last_matched_user = user_id

        if self.consecutive_matches >= self.match_threshold:
            self.confirm_match(user_id, confidence, match_data)
        else:
            self.result_label.config(
                text=f"Verifying match... ({self.consecutive_matches}/{self.match_threshold})",
                foreground='orange'
            )

    def confirm_match(self, user_id, confidence, match_data):
        """Confirm and record successful match"""
        try:
            # Update database
            self.update_match_statistics(user_id)

            # Generate match report
            report = self.generate_match_report(user_id, confidence, match_data)
            
            # Update UI
            self.result_label.config(text=report, foreground='green')
            messagebox.showinfo("Match Result", report)
            self.status_var.set(f"Verified match with '{user_id}'")
            logging.info(f"Verified match: {report}")

        except Exception as e:
            logging.exception(f"Error confirming match: {e}")

    def handle_no_match(self):
        """Handle no match result"""
        self.consecutive_matches = 0
        self.last_matched_user = None
        self.result_label.config(text="No matching user found", foreground='red')
        self.status_var.set("No match found")
        self.progress_bar['value'] = 0  # Reset progress bar

    def handle_matching_error(self, error):
        """Handle matching errors"""
        logging.exception(f"Matching error: {error}", exc_info=True)
        messagebox.showerror("Error", f"Matching failed: {str(error)}")
        self.status_var.set("Matching failed")
        self.progress_bar['value'] = 0  # Reset progress bar

    def generate_match_report(self, user_id, confidence, match_data):
        """Generate detailed match report"""
        knuckle_confidences = {
            k: (v / match_data['total_score'] * confidence)
            for k, v in match_data['scores'].items()
        }
        
        confidence_text = "\n".join(
            f"{knuckle}: {conf:.1f}%" 
            for knuckle, conf in knuckle_confidences.items()
        )
        
        return (f"Verified Match: '{user_id}'\n"
                f"Overall Confidence: {confidence:.1f}%\n"
                f"Individual Confidences:\n{confidence_text}")

    def update_match_statistics(self, user_id):
        """Update match statistics in database"""
        try:
            self.cursor.execute('''
                UPDATE users 
                SET last_match_date = CURRENT_TIMESTAMP,
                    match_count = match_count + 1
                WHERE user_id = ?
            ''', (user_id,))
            self.conn.commit()
        except Exception as e:
            logging.exception(f"Error updating match statistics: {e}")
            raise e

    def init_database(self):
        """Initialize database with enhanced schema and indices"""
        try:
            self.conn = sqlite3.connect('enhanced_knuckle_users.db')
            self.cursor = self.conn.cursor()
            
            # Enable foreign keys
            self.cursor.execute('PRAGMA foreign_keys = ON')
            
            # Create users table with additional metadata
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_match_date TIMESTAMP,
                    match_count INTEGER DEFAULT 0,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    notes TEXT
                )
            ''')
            
            # Create enhanced descriptors table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_descriptors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    knuckle_name TEXT,
                    descriptors BLOB,
                    feature_type TEXT,
                    quality_score REAL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_date TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            # Create indices for better performance
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_descriptors_user_id 
                ON user_descriptors(user_id)
            ''')
            
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_descriptors_quality 
                ON user_descriptors(quality_score)
            ''')
            
            # Create audit log table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT,
                    user_id TEXT,
                    details TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            ''')
            
            self.conn.commit()
            logging.info("Database initialized successfully")
            
        except Exception as e:
            logging.exception(f"Database initialization error: {e}")
            messagebox.showerror("Error", f"Failed to initialize database: {str(e)}")
            raise

    def cleanup_database(self):
        """Database cleanup with corrected query"""
        try:
            # Begin transaction
            self.cursor.execute('BEGIN')
            
            # Update inactive users
            self.cursor.execute('''
                UPDATE users
                SET status = 'inactive'
                WHERE last_match_date < datetime('now', '-90 days')
            ''')
            
            # Delete low quality descriptors
            self.cursor.execute('''
                DELETE FROM user_descriptors
                WHERE quality_score < 50
                AND created_date < datetime('now', '-30 days')
            ''')
            
            # Optimize database
            self.cursor.execute('VACUUM')
            
            # Commit changes
            self.conn.commit()
            logging.info("Database cleanup completed")
            
        except Exception as e:
            self.cursor.execute('ROLLBACK')
            logging.exception(f"Database cleanup error: {e}")
    def backup_database(self):
        """Create backup of the database"""
        try:
            backup_dir = 'backups'
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f'knuckle_db_backup_{timestamp}.db')
            
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path)
            
            # Copy database
            with backup_conn:
                self.conn.backup(backup_conn)
            
            backup_conn.close()
            logging.info(f"Database backed up to {backup_path}")
            
            # Cleanup old backups
            self.cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logging.exception(f"Database backup error: {e}")
            raise

    def cleanup_old_backups(self, backup_dir, keep_days=30):
        """Remove old database backups"""
        try:
            current_time = time.time()
            for backup_file in os.listdir(backup_dir):
                backup_path = os.path.join(backup_dir, backup_file)
                if os.path.getmtime(backup_path) < (current_time - (keep_days * 86400)):
                    os.remove(backup_path)
                    logging.info(f"Removed old backup: {backup_file}")
        except Exception as e:
            logging.exception(f"Backup cleanup error: {e}")

    def log_audit(self, action, user_id=None, details=None):
        """Log audit information"""
        try:
            self.cursor.execute('''
                INSERT INTO audit_log (action, user_id, details)
                VALUES (?, ?, ?)
            ''', (action, user_id, details))
            self.conn.commit()
        except Exception as e:
            logging.exception(f"Audit logging error: {e}")

    def export_user_data(self, user_id):
        """Export user data in a portable format"""
        try:
            export_data = {
                'user_info': {},
                'descriptors': []
            }
            
            # Get user info
            self.cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            user_row = self.cursor.fetchone()
            if user_row:
                export_data['user_info'] = {
                    'user_id': user_row[0],
                    'enrollment_date': user_row[1],
                    'last_match_date': user_row[2],
                    'match_count': user_row[3]
                }
                
                # Get descriptors
                self.cursor.execute('''
                    SELECT knuckle_name, descriptors, feature_type, quality_score
                    FROM user_descriptors
                    WHERE user_id = ?
                ''', (user_id,))
                
                for row in self.cursor.fetchall():
                    export_data['descriptors'].append({
                        'knuckle_name': row[0],
                        'descriptors': row[1],
                        'feature_type': row[2],
                        'quality_score': row[3]
                    })
                
                return export_data
            return None
            
        except Exception as e:
            logging.exception(f"Data export error: {e}")
            raise

    def import_user_data(self, export_data):
        """Import user data from exported format"""
        try:
            if not export_data or 'user_info' not in export_data:
                raise ValueError("Invalid export data format")
                
            user_id = export_data['user_info']['user_id']
            
            # Begin transaction
            self.cursor.execute('BEGIN')
            
            # Insert user
            self.cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, enrollment_date, last_match_date, match_count)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                export_data['user_info']['enrollment_date'],
                export_data['user_info']['last_match_date'],
                export_data['user_info']['match_count']
            ))
            
            # Insert descriptors
            for desc in export_data['descriptors']:
                self.cursor.execute('''
                    INSERT INTO user_descriptors 
                    (user_id, knuckle_name, descriptors, feature_type, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    desc['knuckle_name'],
                    desc['descriptors'],
                    desc['feature_type'],
                    desc['quality_score']
                ))
            
            self.conn.commit()
            self.log_audit("import_user", user_id, "User data imported")
            
        except Exception as e:
            self.cursor.execute('ROLLBACK')
            logging.exception(f"Data import error: {e}")
            raise

    def get_system_statistics(self):
        """Get system usage statistics"""
        try:
            stats = {}
            
            # Get user statistics
            self.cursor.execute('''
                SELECT COUNT(*) as total_users,
                       COUNT(CASE WHEN status = 'active' THEN 1 END) as active_users,
                       AVG(match_count) as avg_matches
                FROM users
            ''')
            row = self.cursor.fetchone()
            stats['user_stats'] = {
                'total_users': row[0],
                'active_users': row[1],
                'average_matches': row[2]
            }
            
            # Get descriptor statistics
            self.cursor.execute('''
                SELECT COUNT(*) as total_descriptors,
                       AVG(quality_score) as avg_quality,
                       AVG(use_count) as avg_uses
                FROM user_descriptors
            ''')
            row = self.cursor.fetchone()
            stats['descriptor_stats'] = {
                'total_descriptors': row[0],
                'average_quality': row[1],
                'average_uses': row[2]
            }
            
            return stats
            
        except Exception as e:
            logging.exception(f"Statistics gathering error: {e}")
            return None

    def on_closing(self):
        """Enhanced application cleanup"""
        try:
            # Stop ongoing processes
            self.stop_webcam()
            
            # Perform database cleanup
            self.cleanup_database()
            
            # Create backup
            self.backup_database()
            
            # Close database connection
            if self.conn:
                self.log_audit("system_shutdown", None, "Application closed")
                self.conn.commit()
                self.conn.close()
                
            logging.info("Application closed successfully")
            
        except Exception as e:
            logging.exception(f"Error during application shutdown: {e}")
            
        finally:
            self.master.destroy()

if __name__ == "__main__":
    try:
        # Configure basic logging first
        logging.basicConfig(
            filename='app.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s'
        )
        
        # Create backup directory
        os.makedirs('backups', exist_ok=True)
        
        # Start application
        root = tk.Tk()
        app = EnhancedKnuckleRecognitionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
        
    except Exception as e:
        # In case of error, try to log to file directly
        with open('error.log', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} CRITICAL: Application failed to start: {str(e)}\n")
        messagebox.showerror("Critical Error", f"Application failed to start: {str(e)}")
