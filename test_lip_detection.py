import cv2
import numpy as np
from utils.lip_detector import LipDetector

def test_lip_detection():
    """Test lip detection functionality"""
    print("Testing Lip Detection...")
    
    # Create lip detector
    lip_detector = LipDetector()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit, 's' to save a test frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        frame_count += 1
        
        # Test lip detection
        try:
            lip_region = lip_detector.detect_lip_region(frame)
            
            if lip_region is not None:
                # Draw bounding box around lip region
                # We'll use a simple rectangle for now
                height, width = frame.shape[:2]
                center_x, center_y = width // 2, height // 2
                lip_size = 100
                
                x1 = max(0, center_x - lip_size // 2)
                y1 = max(0, center_y - lip_size // 2)
                x2 = min(width, center_x + lip_size // 2)
                y2 = min(height, center_y + lip_size // 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Lip Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Lip Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Error in lip detection: {e}")
            cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Lip Detection Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save test frame
            cv2.imwrite('test_frame.jpg', frame)
            print("Test frame saved as 'test_frame.jpg'")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Lip detection test completed!")

if __name__ == "__main__":
    test_lip_detection() 