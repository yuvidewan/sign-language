import cv2
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import sys
sys.path.append('..')

from utils.lip_detector import LipDetector, LipSequenceProcessor
from utils.text_utils import TextProcessor


class LipDataPreprocessor:
    """
    Preprocess lip reading data from videos
    """
    
    def __init__(self, output_size: tuple = (64, 64), sequence_length: int = 30):
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.lip_detector = LipDetector()
        self.processor = LipSequenceProcessor(sequence_length, output_size)
        
    def process_video(self, video_path: str, text: str = None) -> dict:
        """
        Process a single video file
        Args:
            video_path: Path to video file
            text: Corresponding text transcript
        Returns:
            Dictionary with processed data
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        # Extract lip sequences
        lip_sequences = self.lip_detector.extract_lip_sequence(video_path, self.sequence_length)
        
        if not lip_sequences:
            print(f"No lip sequences extracted from: {video_path}")
            return None
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Create output data
        output_data = {
            'video_path': video_path,
            'text': text or "",
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'sequence_length': len(lip_sequences),
            'lip_sequences': lip_sequences
        }
        
        return output_data
    
    def process_directory(self, input_dir: str, output_dir: str, metadata_file: str = None):
        """
        Process all videos in a directory
        Args:
            input_dir: Input directory containing videos
            output_dir: Output directory for processed data
            metadata_file: Optional metadata file with text transcripts
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metadata if provided
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Get video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(video_files)} video files")
        
        # Process each video
        processed_data = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(input_dir, video_file)
            
            # Get corresponding text from metadata
            text = metadata.get(video_file, "")
            
            # Process video
            result = self.process_video(video_path, text)
            
            if result:
                processed_data.append(result)
                
                # Save individual video data
                video_name = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_dir, f"{video_name}.json")
                
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Create dataset metadata
        dataset_metadata = {
            'dataset_info': {
                'total_videos': len(processed_data),
                'output_size': self.output_size,
                'sequence_length': self.sequence_length,
                'processed_date': str(np.datetime64('now'))
            },
            'samples': [
                {
                    'video_path': data['video_path'],
                    'text': data['text'],
                    'duration': data['duration'],
                    'sequence_length': data['sequence_length']
                }
                for data in processed_data
            ]
        }
        
        # Save dataset metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"Processed {len(processed_data)} videos")
        print(f"Dataset metadata saved to: {metadata_path}")
        
        return dataset_metadata
    
    def create_sample_dataset(self, output_dir: str):
        """
        Create a sample dataset for demonstration
        """
        print("Creating sample dataset...")
        
        # Create sample metadata
        sample_metadata = {
            'dataset_info': {
                'total_videos': 10,
                'output_size': self.output_size,
                'sequence_length': self.sequence_length,
                'processed_date': str(np.datetime64('now')),
                'note': 'This is a sample dataset for demonstration'
            },
            'samples': [
                {
                    'video_path': f'sample_video_{i+1}.mp4',
                    'text': f'sample text {i+1}',
                    'duration': 2.5 + i * 0.5,
                    'sequence_length': self.sequence_length
                }
                for i in range(10)
            ]
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        
        print(f"Sample dataset created at: {output_dir}")
        return sample_metadata
    
    def validate_dataset(self, dataset_dir: str) -> dict:
        """
        Validate processed dataset
        Args:
            dataset_dir: Directory containing processed dataset
        Returns:
            Validation report
        """
        metadata_path = os.path.join(dataset_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return {'error': 'Metadata file not found'}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        samples = metadata['samples']
        validation_report = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'invalid_samples': 0,
            'errors': [],
            'statistics': {
                'duration_range': {'min': float('inf'), 'max': 0},
                'text_length_range': {'min': float('inf'), 'max': 0},
                'avg_duration': 0,
                'avg_text_length': 0
            }
        }
        
        total_duration = 0
        total_text_length = 0
        
        for sample in samples:
            try:
                # Check if video file exists
                video_path = sample['video_path']
                if not os.path.exists(video_path):
                    validation_report['errors'].append(f"Video file not found: {video_path}")
                    validation_report['invalid_samples'] += 1
                    continue
                
                # Validate text
                text = sample.get('text', '')
                if not text.strip():
                    validation_report['errors'].append(f"Empty text for video: {video_path}")
                    validation_report['invalid_samples'] += 1
                    continue
                
                # Update statistics
                duration = sample.get('duration', 0)
                text_length = len(text.split())
                
                validation_report['statistics']['duration_range']['min'] = min(
                    validation_report['statistics']['duration_range']['min'], duration
                )
                validation_report['statistics']['duration_range']['max'] = max(
                    validation_report['statistics']['duration_range']['max'], duration
                )
                
                validation_report['statistics']['text_length_range']['min'] = min(
                    validation_report['statistics']['text_length_range']['min'], text_length
                )
                validation_report['statistics']['text_length_range']['max'] = max(
                    validation_report['statistics']['text_length_range']['max'], text_length
                )
                
                total_duration += duration
                total_text_length += text_length
                validation_report['valid_samples'] += 1
                
            except Exception as e:
                validation_report['errors'].append(f"Error processing {sample.get('video_path', 'unknown')}: {str(e)}")
                validation_report['invalid_samples'] += 1
        
        # Calculate averages
        if validation_report['valid_samples'] > 0:
            validation_report['statistics']['avg_duration'] = total_duration / validation_report['valid_samples']
            validation_report['statistics']['avg_text_length'] = total_text_length / validation_report['valid_samples']
        
        return validation_report


def main():
    parser = argparse.ArgumentParser(description='Extract Lip Sequences from Videos')
    parser.add_argument('--video_path', type=str, help='Path to video file or directory')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--metadata_file', type=str, help='Optional metadata file with text transcripts')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--output_size', type=int, nargs=2, default=[64, 64], help='Output size (width height)')
    parser.add_argument('--create_sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--validate', action='store_true', help='Validate existing dataset')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = LipDataPreprocessor(
        output_size=tuple(args.output_size),
        sequence_length=args.sequence_length
    )
    
    if args.create_sample:
        # Create sample dataset
        preprocessor.create_sample_dataset(args.output_path)
        
    elif args.validate:
        # Validate existing dataset
        report = preprocessor.validate_dataset(args.output_path)
        print("Validation Report:")
        print(json.dumps(report, indent=2))
        
    elif args.video_path:
        # Process video(s)
        if os.path.isfile(args.video_path):
            # Single video file
            result = preprocessor.process_video(args.video_path)
            if result:
                print(f"Processed video: {args.video_path}")
                print(f"Duration: {result['duration']:.2f}s")
                print(f"Sequence length: {result['sequence_length']}")
        else:
            # Directory of videos
            preprocessor.process_directory(args.video_path, args.output_path, args.metadata_file)
    else:
        print("Please provide --video_path or use --create_sample or --validate")


if __name__ == '__main__':
    main() 