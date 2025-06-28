import os
import cv2
import torch
from torch.utils.data import Dataset

class GRIDDdataset(Dataset):
    """
    PyTorch Dataset for the GRID corpus.
    Expects directory structure: root/speaker/video/vid1.mpg and root/speaker/align/vid1.align
    Returns (video_frames, transcript) for each sample.
    """
    def __init__(self, root_dir, speakers=None, transform=None, frames_per_clip=75, target_size=(64, 64)):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.target_size = target_size
        self.samples = []
        self._prepare_dataset(speakers)

    def _prepare_dataset(self, speakers):
        speakers = speakers or [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for speaker in speakers:
            video_dir = os.path.join(self.root_dir, speaker, 'video')
            align_dir = os.path.join(self.root_dir, speaker, 'align')
            if not os.path.exists(video_dir) or not os.path.exists(align_dir):
                continue
            for fname in os.listdir(video_dir):
                if fname.endswith('.mpg'):
                    video_path = os.path.join(video_dir, fname)
                    base = os.path.splitext(fname)[0]
                    transcript_path = os.path.join(align_dir, base + '.align')
                    if os.path.exists(transcript_path):
                        self.samples.append((video_path, transcript_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, transcript_path = self.samples[idx]
        frames = self._load_video(video_path)
        transcript = self._load_transcript(transcript_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        return frames, transcript

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to target size
            frame = cv2.resize(frame, self.target_size)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
            count += 1
            if count >= self.frames_per_clip:
                break
        cap.release()
        # Pad or trim to frames_per_clip
        if len(frames) < self.frames_per_clip:
            pad = [frames[-1]] * (self.frames_per_clip - len(frames))
            frames.extend(pad)
        else:
            frames = frames[:self.frames_per_clip]
        return frames

    def _load_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # .align files: each line is 'start end word', last line is 'sil'
        words = [line.strip().split()[-1] for line in lines if line.strip()]
        # Remove 'sil' (silence) tokens
        words = [w for w in words if w.lower() != 'sil']
        return ' '.join(words) 