import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List
import torch
from PIL import Image


class LipDetector:
    """
    Lip detection and extraction using MediaPipe Face Mesh
    """
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (MediaPipe Face Mesh)
        self.lip_landmarks = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 321,
            375, 307, 320, 405, 314, 17, 84, 61
        ]
        
        # Additional landmarks for better lip region
        self.outer_lip_landmarks = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 308, 324, 318, 78, 95, 88, 178,
            87, 14, 317, 402, 318, 324, 308, 321, 375, 307,
            320, 405, 314, 17, 84, 61
        ]
        
    def detect_lip_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract lip region from frame
        Args:
            frame: Input frame (BGR format)
        Returns:
            Cropped lip region or None if not detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract lip landmarks
        lip_points = []
        height, width = frame.shape[:2]
        
        for landmark_id in self.outer_lip_landmarks:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            lip_points.append([x, y])
        
        lip_points = np.array(lip_points, dtype=np.int32)
        
        # Get bounding box with padding
        x, y, w, h = cv2.boundingRect(lip_points)
        
        # Add padding to include more context
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        # Crop lip region
        lip_region = frame[y:y+h, x:x+w]
        
        if lip_region.size == 0:
            return None
        
        return lip_region
    
    def extract_lip_landmarks(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Extract lip landmark coordinates
        Args:
            frame: Input frame
        Returns:
            List of (x, y) coordinates or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        height, width = frame.shape[:2]
        
        landmarks = []
        for landmark_id in self.lip_landmarks:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))
        
        return landmarks
    
    def draw_lip_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw lip landmarks on frame
        Args:
            frame: Input frame
            landmarks: List of (x, y) coordinates
        Returns:
            Frame with landmarks drawn
        """
        frame_copy = frame.copy()
        
        for x, y in landmarks:
            cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)
        
        # Draw lip contour
        if len(landmarks) > 2:
            points = np.array(landmarks, dtype=np.int32)
            cv2.polylines(frame_copy, [points], True, (0, 255, 0), 2)
        
        return frame_copy
    
    def preprocess_lip_region(self, lip_region: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        """
        Preprocess lip region for model input
        Args:
            lip_region: Cropped lip region
            target_size: Target size (height, width)
        Returns:
            Preprocessed tensor
        """
        # Resize to target size
        resized = cv2.resize(lip_region, target_size)
        
        # Convert to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)  # (C, H, W)
        
        return tensor
    
    def extract_lip_sequence(self, video_path: str, max_frames: int = 100) -> List[torch.Tensor]:
        """
        Extract lip sequence from video
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        Returns:
            List of preprocessed lip tensors
        """
        cap = cv2.VideoCapture(video_path)
        lip_sequences = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            lip_region = self.detect_lip_region(frame)
            if lip_region is not None:
                processed = self.preprocess_lip_region(lip_region)
                lip_sequences.append(processed)
                frame_count += 1
        
        cap.release()
        return lip_sequences
    
    def real_time_lip_detection(self, callback=None):
        """
        Real-time lip detection from webcam
        Args:
            callback: Function to call with detected lip region
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect lip region
            lip_region = self.detect_lip_region(frame)
            
            if lip_region is not None:
                # Draw lip region on frame
                x, y, w, h = cv2.boundingRect(lip_region)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Call callback if provided
                if callback:
                    callback(lip_region)
            
            # Show frame
            cv2.imshow('Lip Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


class LipSequenceProcessor:
    """
    Process lip sequences for training and inference
    """
    
    def __init__(self, sequence_length: int = 30, target_size: Tuple[int, int] = (64, 64)):
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.lip_detector = LipDetector()
    
    def process_video_sequence(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Process video into lip sequence tensor
        Args:
            video_path: Path to video file
        Returns:
            Tensor of shape (sequence_length, channels, height, width) or None
        """
        lip_sequences = self.lip_detector.extract_lip_sequence(video_path, self.sequence_length)
        
        if len(lip_sequences) == 0:
            return None
        
        # Pad or truncate to sequence_length
        if len(lip_sequences) < self.sequence_length:
            # Pad with last frame
            last_frame = lip_sequences[-1]
            while len(lip_sequences) < self.sequence_length:
                lip_sequences.append(last_frame)
        else:
            # Truncate to sequence_length
            lip_sequences = lip_sequences[:self.sequence_length]
        
        # Stack into tensor
        sequence_tensor = torch.stack(lip_sequences)
        
        return sequence_tensor
    
    def create_batch(self, video_paths: List[str]) -> torch.Tensor:
        """
        Create batch from multiple video paths
        Args:
            video_paths: List of video file paths
        Returns:
            Batch tensor of shape (batch_size, sequence_length, channels, height, width)
        """
        sequences = []
        
        for video_path in video_paths:
            sequence = self.process_video_sequence(video_path)
            if sequence is not None:
                sequences.append(sequence)
        
        if not sequences:
            return torch.empty(0)
        
        # Stack into batch
        batch = torch.stack(sequences)
        
        return batch 