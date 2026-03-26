# ============================================
# CELL 1: Install Dependencies
# ============================================
"""
!pip install -q gradio ultralytics transformers torch opencv-python-headless pillow numpy twilio fastapi uvicorn
!apt-get update -qq && apt-get install -y -qq aria2
"""

# ============================================
# CELL 2: Complete Application Code
# ============================================

import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import time
import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from datetime import datetime
from twilio.rest import Client

# ============================================
# Dummy Backend & Twilio Integration
# ============================================

class MockRTOBackend:
    """Mock API to fetch vehicle owner details using license plate."""
    @staticmethod
    def get_owner_details(plate_number: str) -> dict:
        # Returns dummy data for demonstration
        return {
            "owner_name": "Ayush Nirmal",
            "vehicle_model": "Honda Activa",
            "registration_date": "2023-05-12"
        }

class TwilioAlertSystem:
    """Handles sending SMS via Twilio"""
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.is_configured = bool(account_sid and auth_token and from_number)
        
        if self.is_configured:
            try:
                self.client = Client(account_sid, auth_token)
            except Exception as e:
                print(f"Twilio Configuration Error: {e}")
                self.is_configured = False
        else:
            self.client = None
            
    def send_violation_alert(self, plate_number: str, owner_details: dict, to_number: str) -> str:
        if not to_number:
            return "No destination number provided."
            
        message_body = (
            f"Hey {owner_details.get('owner_name', 'Unknown')},\n"
            f"Vehicle {plate_number} ({owner_details.get('vehicle_model', 'Unknown')}) registered under your name.\n"
            f"You have been found to have not wearing helmet in Junction Number 7 CBS circle."
        )
        
        if not self.is_configured:
            print(f"[MOCK SMS to {to_number}]\n{message_body}")
            return f"SMS configuration missing. Mock message generated."
            
        try:
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=to_number
            )
            print(f"[TWILIO SMS SENT] SID: {message.sid}")
            return f"SMS Sent successfully: {message.sid}"
        except Exception as e:
            print(f"[TWILIO ERROR] {e}")
            return f"SMS Failed: {e}"

# ============================================
# Configuration & Data Classes
# ============================================

@dataclass
class DetectionConfig:
    """Configuration for detection parameters"""
    # YOLO Models (Public access URLs via PyTorch Fallback or aria2c)
    HELMET_MODEL_URL: str = "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt"
    PLATE_MODEL_URL: str = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
    VEHICLE_MODEL_URL: str = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    
    # OCR Model (TrOCR for plate reading)
    OCR_PROCESSOR: str = "microsoft/trocr-base-printed"
    OCR_MODEL: str = "microsoft/trocr-base-printed"
    
    # Local paths
    MODEL_DIR: str = "models"
    OUTPUT_DIR: str = "outputs"
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    MAX_WORKERS: int = 4
    
    # Class mappings
    HELMET_CLASSES = {0: 'head', 1: 'helmet', 2: 'person'}
    PLATE_CLASSES = {0: 'license_plate'}
    VEHICLE_CLASSES = {
        0: 'bus', 1: 'car', 2: 'minibus', 3: 'truck', 
        4: 'van', 5: 'motorcycle', 6: 'bicycle'
    }

# ============================================
# Aria2c Download Manager
# ============================================

class Aria2cDownloader:
    """High-speed downloader using aria2c with multithreading"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.download_queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        
    def download_file(self, url: str, output_path: str, filename: str = None) -> str:
        """
        Download file using aria2c with optimized settings
        """
        if filename is None:
            filename = os.path.basename(url)
        
        output_file = os.path.join(output_path, filename)
        
        # Skip if already exists
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            print(f"✓ {filename} already exists")
            return output_file
        
        os.makedirs(output_path, exist_ok=True)
        
        # aria2c command with optimal settings for Hugging Face
        cmd = [
            'aria2c',
            '--console-log-level=error',
            '--summary-interval=0',
            '--max-connection-per-server=16',
            '--split=16',
            '--min-split-size=1M',
            '--max-concurrent-downloads=8',
            '--file-allocation=none',
            '--check-certificate=false',
            '--allow-overwrite=true',
            '--auto-file-renaming=false',
            '-d', output_path,
            '-o', filename,
            url
        ]
        
        try:
            print(f"⬇️ Downloading {filename}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_file):
                size = os.path.getsize(output_file) / (1024 * 1024)
                print(f"✓ Downloaded {filename} ({size:.2f} MB) via aria2c")
                return output_file
            else:
                raise Exception(f"Download failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error downloading {filename} via aria2c: {e}")
            print(f"🔄 Using PyTorch fallback download for {filename}...")
            # Fallback to PyTorch download (handles redirects and user agents)
            try:
                import torch
                torch.hub.download_url_to_file(url, output_file, progress=False)
                if os.path.exists(output_file):
                    size = os.path.getsize(output_file) / (1024 * 1024)
                    print(f"✓ Downloaded {filename} ({size:.2f} MB) via fallback")
                    return output_file
                else:
                    raise Exception("File not found after download.")
            except Exception as fe:
                print(f"❌ Fallback download failed for {filename}: {fe}")
                raise

    def download_multiple(self, downloads: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Download multiple files concurrently
        downloads: List of dicts with 'url', 'output_path', 'filename'
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_file = {
                executor.submit(
                    self.download_file, 
                    d['url'], 
                    d['output_path'], 
                    d.get('filename')
                ): d.get('filename', os.path.basename(d['url']))
                for d in downloads
            }
            
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results[filename] = result
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
                    results[filename] = None
                    
        return results

# ============================================
# Model Manager
# ============================================

class ModelManager:
    """Manages YOLO and Hugging Face models with lazy loading"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.downloader = Aria2cDownloader()
        
        # Model storage
        self.helmet_model: Optional[YOLO] = None
        self.plate_model: Optional[YOLO] = None
        self.vehicle_model: Optional[YOLO] = None
        self.ocr_processor = None
        self.ocr_model = None
        
        # Thread safety
        self.model_lock = threading.Lock()
        self.loading_status = {}
        
    def initialize(self, progress_callback=None):
        """Download and load all models"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        # Define downloads
        downloads = [
            {
                'url': self.config.HELMET_MODEL_URL,
                'output_path': self.config.MODEL_DIR,
                'filename': 'helmet_model.pt'
            },
            {
                'url': self.config.PLATE_MODEL_URL,
                'output_path': self.config.MODEL_DIR,
                'filename': 'plate_model.pt'
            },
            {
                'url': self.config.VEHICLE_MODEL_URL,
                'output_path': self.config.MODEL_DIR,
                'filename': 'vehicle_model.pt'
            }
        ]
        
        # Download all models concurrently
        if progress_callback:
            progress_callback(0.1, "Downloading YOLO models...")
        
        downloaded = self.downloader.download_multiple(downloads)
        
        if progress_callback:
            progress_callback(0.4, "Loading YOLO models...")
        
        # Load YOLO models
        with self.model_lock:
            self.helmet_model = YOLO(downloaded['helmet_model.pt'], verbose=False)
            self.plate_model = YOLO(downloaded['plate_model.pt'], verbose=False)
            self.vehicle_model = YOLO(downloaded['vehicle_model.pt'], verbose=False)
            
            # Dynamically update class mappings based on model configuration
            if hasattr(self.helmet_model, 'names'):
                self.config.HELMET_CLASSES = self.helmet_model.names
            if hasattr(self.plate_model, 'names'):
                self.config.PLATE_CLASSES = self.plate_model.names
            if hasattr(self.vehicle_model, 'names'):
                self.config.VEHICLE_CLASSES = self.vehicle_model.names
        
        if progress_callback:
            progress_callback(0.7, "Loading OCR model...")
        
        # Load Hugging Face OCR model
        self.ocr_processor = TrOCRProcessor.from_pretrained(self.config.OCR_PROCESSOR)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.OCR_MODEL)
        
        if torch.cuda.is_available():
            self.ocr_model = self.ocr_model.cuda()
        
        if progress_callback:
            progress_callback(1.0, "Models ready!")
        
        return True

    def get_models(self):
        """Thread-safe model access"""
        with self.model_lock:
            return {
                'helmet': self.helmet_model,
                'plate': self.plate_model,
                'vehicle': self.vehicle_model,
                'ocr_processor': self.ocr_processor,
                'ocr_model': self.ocr_model
            }

# ============================================
# Detection Engine
# ============================================

class DetectionEngine:
    """Core detection logic with multithreading"""
    
    def __init__(self, model_manager: ModelManager, config: DetectionConfig):
        self.model_manager = model_manager
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
    def detect_helmets(self, image: np.ndarray) -> List[Dict]:
        """
        Detect helmets and heads in image
        Returns: List of detections with bbox, class, confidence
        """
        model = self.model_manager.get_models()['helmet']
        results = model.predict(
            image, 
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            cls_name = self.config.HELMET_CLASSES.get(cls, 'unknown').lower()
            if 'without' in cls_name or 'no helmet' in cls_name or 'head' in cls_name:
                mapped_class = 'head'
            elif 'helmet' in cls_name or 'with helmet' in cls_name:
                mapped_class = 'helmet'
            else:
                mapped_class = 'person'
                
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': mapped_class,
                'confidence': conf,
                'type': 'helmet'
            })
            
        return detections

    def detect_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates in image
        """
        model = self.model_manager.get_models()['plate']
        results = model.predict(
            image,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop plate for OCR
            plate_crop = image[y1:y2, x1:x2]
            plate_text = self.read_plate(plate_crop) if plate_crop.size > 0 else ""
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': 'license_plate',
                'confidence': conf,
                'plate_text': plate_text,
                'type': 'plate'
            })
            
        return detections

    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in image
        """
        model = self.model_manager.get_models()['vehicle']
        results = model.predict(
            image,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': self.config.VEHICLE_CLASSES.get(cls, 'vehicle'),
                'confidence': conf,
                'type': 'vehicle'
            })
            
        return detections

    def read_plate(self, plate_image: np.ndarray) -> str:
        """
        OCR for license plate using TrOCR
        """
        try:
            if plate_image.size == 0:
                return ""
                
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            
            # Prepare for model
            processor = self.model_manager.get_models()['ocr_processor']
            model = self.model_manager.get_models()['ocr_model']
            
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            # Generate
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean text
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            return text
            
        except Exception as e:
            return ""

    def detect_all(self, image: np.ndarray, parallel: bool = True) -> Dict:
        """
        Run all detections (parallel or sequential)
        """
        if parallel:
            # Submit all tasks concurrently
            future_helmet = self.executor.submit(self.detect_helmets, image)
            future_plate = self.executor.submit(self.detect_plates, image)
            future_vehicle = self.executor.submit(self.detect_vehicles, image)
            
            helmets = future_helmet.result()
            plates = future_plate.result()
            vehicles = future_vehicle.result()
        else:
            helmets = self.detect_helmets(image)
            plates = self.detect_plates(image)
            vehicles = self.detect_vehicles(image)
        
        # Associate helmets with vehicles
        helmets = self._associate_helmets_with_vehicles(helmets, vehicles)
        
        return {
            'helmets': helmets,
            'plates': plates,
            'vehicles': vehicles,
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(helmets, plates, vehicles)
        }

    def _associate_helmets_with_vehicles(self, helmets: List[Dict], vehicles: List[Dict]) -> List[Dict]:
        """Associate helmet detections with vehicle detections"""
        for helmet in helmets:
            helmet['associated_vehicle'] = None
            hx1, hy1, hx2, hy2 = helmet['bbox']
            h_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
            
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['bbox']
                # Check if helmet center is inside vehicle bbox
                if vx1 <= h_center[0] <= vx2 and vy1 <= h_center[1] <= vy2:
                    helmet['associated_vehicle'] = vehicle['class']
                    break
                    
        return helmets

    def _generate_summary(self, helmets: List[Dict], plates: List[Dict], vehicles: List[Dict]) -> Dict:
        """Generate detection summary"""
        helmet_count = sum(1 for h in helmets if h['class'] == 'helmet')
        no_helmet_count = sum(1 for h in helmets if h['class'] == 'head')
        
        return {
            'total_vehicles': len(vehicles),
            'total_helmets_detected': helmet_count,
            'total_no_helmet': no_helmet_count,
            'total_plates': len(plates),
            'vehicles_with_no_helmet': len(set(
                h.get('associated_vehicle') for h in helmets 
                if h['class'] == 'head' and h.get('associated_vehicle')
            )),
            'compliance_rate': helmet_count / (helmet_count + no_helmet_count) * 100 
                              if (helmet_count + no_helmet_count) > 0 else 0
        }

# ============================================
# Visualization
# ============================================

class Visualizer:
    """Handle visualization of detection results"""
    
    COLORS = {
        'helmet': (0, 255, 0),      # Green
        'head': (0, 0, 255),        # Red
        'person': (255, 255, 0),    # Cyan
        'license_plate': (255, 0, 255),  # Magenta
        'vehicle': (255, 165, 0),   # Orange
        'bus': (128, 0, 128),       # Purple
        'car': (0, 128, 255),       # Light Blue
        'truck': (0, 69, 255),      # Dark Orange
        'motorcycle': (128, 128, 0), # Olive
        'text_bg': (0, 0, 0)
    }
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def draw_detections(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Draw all detections on image"""
        img = image.copy()
        
        # Draw vehicles first (bottom layer)
        for det in results['vehicles']:
            img = self._draw_bbox(img, det, 'vehicle')
            
        # Draw helmets
        for det in results['helmets']:
            img = self._draw_bbox(img, det, det['class'])
            
        # Draw plates
        for det in results['plates']:
            img = self._draw_bbox(img, det, 'license_plate', 
                                  extra_text=det.get('plate_text', ''))
        
        # Draw summary panel
        img = self._draw_summary_panel(img, results['summary'])
        
        return img

    def _draw_bbox(self, img: np.ndarray, detection: Dict, 
                   color_key: str, extra_text: str = "") -> np.ndarray:
        """Draw single bounding box"""
        x1, y1, x2, y2 = detection['bbox']
        color = self.COLORS.get(color_key, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)
        
        # Prepare label
        label = f"{detection['class']} {detection['confidence']:.2f}"
        if extra_text:
            label += f" | {extra_text}"
        if detection.get('associated_vehicle'):
            label += f" ({detection['associated_vehicle']})"
            
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, self.thickness)
        
        # Draw text
        cv2.putText(img, label, (x1, y1 - 5), self.font, self.font_scale, 
                   (255, 255, 255), 1, cv2.LINE_AA)
        
        return img

    def _draw_summary_panel(self, img: np.ndarray, summary: Dict) -> np.ndarray:
        """Draw detection summary panel"""
        h, w = img.shape[:2]
        panel_w = 350
        panel_h = 180
        x = w - panel_w - 20
        y = 20
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # Border
        cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(img, "DETECTION SUMMARY", (x + 10, y + 30), 
                   self.font, 0.7, (0, 255, 255), 2)
        
        # Stats
        stats = [
            f"Vehicles: {summary['total_vehicles']}",
            f"Helmets: {summary['total_helmets_detected']}",
            f"No Helmet: {summary['total_no_helmet']}",
            f"Plates: {summary['total_plates']}",
            f"Compliance: {summary['compliance_rate']:.1f}%"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(img, stat, (x + 10, y + 60 + i * 25), 
                       self.font, 0.6, (255, 255, 255), 1)
        
        return img

    def create_comparison(self, original: np.ndarray, detected: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison"""
        h, w = original.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w:] = detected
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), self.font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Detected", (w + 10, 30), self.font, 1, (0, 255, 0), 2)
        
        return combined

# ============================================
# Video Processor
# ============================================

class VideoProcessor:
    """Handle video processing with multithreading"""
    
    def __init__(self, engine: DetectionEngine, visualizer: Visualizer):
        self.engine = engine
        self.visualizer = visualizer
        self.frame_buffer = queue.Queue(maxsize=30)
        self.result_buffer = queue.Queue(maxsize=30)
        self.processing = False
        
    def process_video(self, video_path: str, progress_callback=None, twilio_system=None, twilio_to_number=None) -> Tuple[str, str]:
        """
        Process video file with threaded frame processing AND Twilio Alert Integration
        """
        sms_status_msg = "No SMS triggered."
        alert_sent = False
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output path
        output_dir = self.engine.config.OUTPUT_DIR
        output_path = os.path.join(output_dir, f"processed_{int(time.time())}.mp4")
        os.makedirs(output_dir, exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame for speed (adjust as needed)
            if frame_count % 2 == 0:  # Process every 2nd frame
                results = self.engine.detect_all(frame, parallel=True)
                vis_frame = self.visualizer.draw_detections(frame, results)
                
                # Check for helmet violation and visible plate for SMS Alert
                if twilio_system and twilio_to_number and not alert_sent:
                    # Get violations (head = no helmet) and plates
                    violations = [h for h in results['helmets'] if h['class'] == 'head']
                    plates = [p for p in results['plates'] if p.get('plate_text')]
                    
                    if violations and plates:
                        plate_number = plates[0]['plate_text']
                        # Fetch from Mock API
                        owner_details = MockRTOBackend.get_owner_details(plate_number)
                        # Send alert
                        sms_status_msg = twilio_system.send_violation_alert(
                            plate_number, owner_details, twilio_to_number
                        )
                        alert_sent = True
                        if progress_callback:
                            progress_callback(min(1.0, frame_count / total_frames), f"Violation SMS Sent: {plate_number}")
            else:
                vis_frame = frame
                
            out.write(vis_frame)
            processed_count += 1
            
            if progress_callback and frame_count % 10 == 0:
                progress = min(1.0, frame_count / total_frames)
                progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
                
            frame_count += 1
            
        cap.release()
        out.release()
        
        return output_path, sms_status_msg

# ============================================
# Gradio Interface
# ============================================

class HelmetDetectionApp:
    """Main Gradio Application"""
    
    def __init__(self):
        self.config = DetectionConfig()
        self.model_manager = ModelManager(self.config)
        self.engine = None
        self.visualizer = Visualizer()
        self.video_processor = None
        self.initialized = False
        
    def initialize_models(self, progress=gr.Progress()):
        """Initialize all models with progress tracking"""
        if self.initialized:
            return "Models already loaded!"
            
        def update_progress(val, msg):
            progress(val, desc=msg)
            
        try:
            self.model_manager.initialize(update_progress)
            self.engine = DetectionEngine(self.model_manager, self.config)
            self.video_processor = VideoProcessor(self.engine, self.visualizer)
            self.initialized = True
            return "✅ All models loaded successfully!"
        except Exception as e:
            return f"❌ Error loading models: {str(e)}"
    
    def process_image(self, image: np.ndarray, parallel: bool = True) -> Tuple[np.ndarray, str]:
        """Process single image"""
        if not self.initialized:
            return image, "Please initialize models first!"
            
        if image is None:
            return None, "No image provided"
            
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Run detection
        start_time = time.time()
        results = self.engine.detect_all(image, parallel=parallel)
        process_time = time.time() - start_time
        
        # Visualize
        vis_image = self.visualizer.draw_detections(image, results)
        
        # Convert back to RGB for Gradio
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Generate report
        report = self._generate_text_report(results, process_time)
        
        return vis_image, report

    def process_video(self, video_path: str, acc_sid: str, auth_token: str, from_num: str, to_num: str, progress=gr.Progress()) -> Tuple[str, str, str]:
        """Process video file"""
        if not self.initialized:
            return None, "Please initialize models first!", "No SMS Sent"
            
        if video_path is None:
            return None, "No video provided", "No SMS Sent"
            
        def update_progress(val, msg):
            progress(val, desc=msg)
            
        # Initialize Twilio for this run
        twilio_sys = TwilioAlertSystem(acc_sid, auth_token, from_num)
            
        output_path, sms_status = self.video_processor.process_video(
            video_path, update_progress, twilio_sys, to_num
        )
        
        return output_path, f"Video processed successfully! Saved to: {output_path}", sms_status

    def process_batch(self, files: List[str]) -> str:
        """Process multiple images"""
        if not self.initialized:
            return "Please initialize models first!"
            
        if not files:
            return "No files provided"
            
        results_summary = []
        
        for i, file_path in enumerate(files):
            try:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                    
                results = self.engine.detect_all(img, parallel=True)
                summary = results['summary']
                
                results_summary.append({
                    'file': os.path.basename(file_path),
                    'vehicles': summary['total_vehicles'],
                    'helmets': summary['total_helmets_detected'],
                    'no_helmet': summary['total_no_helmet'],
                    'compliance': f"{summary['compliance_rate']:.1f}%"
                })
            except Exception as e:
                results_summary.append({
                    'file': os.path.basename(file_path),
                    'error': str(e)
                })
                
        # Format results
        output = "BATCH PROCESSING RESULTS\n" + "="*50 + "\n\n"
        for r in results_summary:
            if 'error' in r:
                output += f"❌ {r['file']}: ERROR - {r['error']}\n\n"
            else:
                output += f"📄 {r['file']}\n"
                output += f"   Vehicles: {r['vehicles']} | "
                output += f"Helmets: {r['helmets']} | "
                output += f"No Helmet: {r['no_helmet']} | "
                output += f"Compliance: {r['compliance']}\n\n"
                
        return output

    def _generate_text_report(self, results: Dict, process_time: float) -> str:
        """Generate detailed text report"""
        s = results['summary']
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           HELMET & PLATE DETECTION REPORT               ║
╠══════════════════════════════════════════════════════════╣
  Processing Time: {process_time:.3f}s
  Timestamp: {results['timestamp']}
  
┌─────────────────────────────────────────────────────────┐
│ DETECTION SUMMARY                                       │
├─────────────────────────────────────────────────────────┤
  • Total Vehicles Detected: {s['total_vehicles']}
  • Persons with Helmets: {s['total_helmets_detected']}
  • Persons without Helmets: {s['total_no_helmet']} ⚠️
  • License Plates Found: {s['total_plates']}
  • Safety Compliance Rate: {s['compliance_rate']:.1f}%
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DETECTED OBJECTS                                        │
├─────────────────────────────────────────────────────────┤"""
        
        if results['vehicles']:
            report += "\n  VEHICLES:\n"
            for v in results['vehicles']:
                report += f"    └─ {v['class'].upper()} (conf: {v['confidence']:.2f})\n"
                
        if results['helmets']:
            report += "\n  HELMET STATUS:\n"
            for h in results['helmets']:
                status = "✅ HELMET" if h['class'] == 'helmet' else "❌ NO HELMET"
                vehicle = f" on {h['associated_vehicle']}" if h.get('associated_vehicle') else ""
                report += f"    └─ {status}{vehicle} (conf: {h['confidence']:.2f})\n"
                
        if results['plates']:
            report += "\n  LICENSE PLATES:\n"
            for p in results['plates']:
                text = p.get('plate_text', 'N/A')
                report += f"    └─ [{text}] (conf: {p['confidence']:.2f})\n"
                
        report += "\n└─────────────────────────────────────────────────────────┘\n"
        
        # Violations
        violations = [h for h in results['helmets'] if h['class'] == 'head']
        if violations:
            report += "\n⚠️  SAFETY VIOLATIONS DETECTED:\n"
            for v in violations:
                vehicle = v.get('associated_vehicle', 'unknown vehicle')
                report += f"   • Person without helmet on {vehicle}\n"
                
        return report

    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="🪖 Helmet & Vehicle Detection System", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🪖 Helmet & Vehicle Number Plate Detection System
            
            **Powered by:** YOLOv11 + Hugging Face Models + aria2c + Multithreading
            
            ### Features:
            - ✅ Helmet detection (worn/not worn)
            - ✅ Vehicle detection & classification  
            - ✅ License plate detection & OCR
            - ✅ Safety compliance reporting
            - ✅ Parallel processing support
            """)
            
            with gr.Tab("⚙️ Setup"):
                gr.Markdown("### Step 1: Initialize Models")
                gr.Markdown("Downloads YOLOv11 models using aria2c (high-speed) and loads Hugging Face OCR models")
                
                init_btn = gr.Button("🚀 Initialize Models", variant="primary")
                init_status = gr.Textbox(label="Status", value="Not initialized")
                
                init_btn.click(
                    fn=self.initialize_models,
                    outputs=init_status
                )
            
            with gr.Tab("🖼️ Image Detection"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="numpy")
                        parallel_check = gr.Checkbox(
                            label="Use Parallel Processing", 
                            value=True,
                            info="Faster but uses more GPU memory"
                        )
                        detect_btn = gr.Button("🔍 Detect", variant="primary")
                        
                    with gr.Column():
                        output_image = gr.Image(label="Detection Result")
                        output_report = gr.Textbox(
                            label="Detection Report", 
                            lines=20,
                            max_lines=30
                        )
                
                detect_btn.click(
                    fn=self.process_image,
                    inputs=[input_image, parallel_check],
                    outputs=[output_image, output_report]
                )
                
                # Examples
                gr.Examples(
                    examples=["sample_image.jpg"] if os.path.exists("sample_image.jpg") else [],
                    inputs=input_image,
                    label="Example Images (add your own to this folder)"
                )
            
            with gr.Tab("🎬 Video Detection"):
                gr.Markdown("### Process Video Files")
                gr.Markdown("Note: Video processing skips every other frame for speed")
                
                input_video = gr.Video(label="Upload Video")
                
                with gr.Accordion("📱 Twilio SMS Settings (Optional)", open=False):
                    tw_acc_sid = gr.Textbox(label="Twilio Account SID", type="password")
                    tw_auth_token = gr.Textbox(label="Twilio Auth Token", type="password")
                    tw_from_num = gr.Textbox(label="Twilio From Number (e.g. +1234567890)")
                    tw_to_num = gr.Textbox(label="Destination Phone Number (e.g. +0987654321)")
                
                process_video_btn = gr.Button("🎥 Process Video", variant="primary")
                
                with gr.Row():
                    output_video = gr.Video(label="Processed Video")
                    video_status = gr.Textbox(label="Status")
                    sms_status_ui = gr.Textbox(label="SMS Alert Status")
                
                process_video_btn.click(
                    fn=self.process_video,
                    inputs=[input_video, tw_acc_sid, tw_auth_token, tw_from_num, tw_to_num],
                    outputs=[output_video, video_status, sms_status_ui]
                )
            
            with gr.Tab("📁 Batch Processing"):
                gr.Markdown("### Process Multiple Images")
                
                batch_files = gr.File(
                    label="Upload Multiple Images", 
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_btn = gr.Button("📊 Process Batch", variant="primary")
                batch_output = gr.Textbox(label="Batch Results", lines=25)
                
                batch_btn.click(
                    fn=self.process_batch,
                    inputs=batch_files,
                    outputs=batch_output
                )
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## Model Information
                
                ### YOLOv11 Models (via Hugging Face):
                - **Helmet Detection:** `keremberke/yolov11m-helmet-detection`
                - **License Plate:** `keremberke/yolov11n-license-plate`
                - **Vehicle Detection:** `keremberke/yolov11m-vehicle-detection`
                
                ### OCR Model:
                - **TrOCR:** `microsoft/trocr-base-printed`
                
                ### Technical Stack:
                - **Download Manager:** aria2c (16 connections, multithreaded)
                - **Inference:** PyTorch with CUDA support
                - **UI:** Gradio 4.x
                - **Processing:** ThreadPoolExecutor for parallel detection
                
                ### Detection Classes:
                - Helmet, Head, Person
                - License Plate (with text recognition)
                - Car, Bus, Truck, Van, Motorcycle, Minibus, Bicycle
                
                ### Performance Tips:
                1. Use parallel processing for images (3x speedup)
                2. For videos, every 2nd frame is processed
                3. Models are cached after first download
                4. Adjust confidence threshold in code if needed
                """)
        
        return demo

# ============================================
# Main Execution
# ============================================

def main():
    """Main entry point"""
    app = HelmetDetectionApp()
    demo = app.create_interface()
    
    # Launch with Colab-friendly settings
    demo.launch(
        share=True,  # Creates public URL for Colab
        debug=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    main()
