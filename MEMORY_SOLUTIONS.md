# Memory Solutions for Lip Reading Training

## Problem
You encountered a memory error when training the lip reading model:
```
RuntimeError: [enforce fail at alloc_cpu.cpp:116] data. DefaultCPUAllocator: not enough memory: you tried to allocate 7962624000 bytes.
```

This happens because the model is trying to allocate ~8GB of memory, which exceeds your available RAM.

## Root Cause
The memory issue occurs due to:
1. **Large input data**: 75 frames × 3 channels × 64×64 pixels × 4 samples = ~14.7 MB just for input
2. **Intermediate activations**: CNN layers create much larger intermediate tensors
3. **Batch size**: Processing multiple samples simultaneously multiplies memory usage
4. **Model complexity**: The original model has 4 CNN layers, bidirectional LSTM, and attention mechanism

## Solutions

### 1. Immediate Fix - Use Reduced Parameters

Try the original model with reduced parameters:

```bash
python -m training.train_lip_reader --data_path data/GRID --epochs 10 --batch_size 1 --frames_per_clip 30 --device cpu --train_speakers s1 s2 s3
```

**Key changes:**
- `--batch_size 1` (reduced from 4)
- `--frames_per_clip 30` (reduced from 75)
- `--device cpu` (if GPU memory is insufficient)

### 2. Use the Memory-Optimized Training Script

```bash
python -m training.train_lip_reader_memory_optimized --data_path data/GRID --epochs 10 --batch_size 1 --frames_per_clip 30 --device cpu --train_speakers s1 s2 s3
```

This script includes:
- Gradient checkpointing
- Memory cleanup after each batch
- Error handling for OOM errors
- Reduced default parameters

### 3. Use the Lightweight Model (Recommended)

```bash
python -m training.train_lightweight_lip_reader --data_path data/GRID --epochs 10 --batch_size 2 --frames_per_clip 40 --device cpu --train_speakers s1 s2 s3
```

The lightweight model:
- Uses 3 CNN layers instead of 4
- Single-layer unidirectional LSTM (instead of 2-layer bidirectional)
- No attention mechanism
- Smaller hidden size (128 vs 256)
- Can handle larger batch sizes and more frames

### 4. Test Memory Usage First

Run the memory test script to find optimal parameters for your system:

```bash
python test_memory_usage.py
```

This will test different configurations and recommend the best parameters.

## Parameter Recommendations

### For Systems with Limited RAM (8GB or less):
```bash
# Original model
python -m training.train_lip_reader --batch_size 1 --frames_per_clip 25 --device cpu --num_workers 0

# Lightweight model (better)
python -m training.train_lightweight_lip_reader --batch_size 1 --frames_per_clip 35 --device cpu --num_workers 0
```

### For Systems with Moderate RAM (16GB):
```bash
# Original model
python -m training.train_lip_reader --batch_size 2 --frames_per_clip 30 --device cpu --num_workers 0

# Lightweight model
python -m training.train_lightweight_lip_reader --batch_size 2 --frames_per_clip 40 --device cpu --num_workers 0
```

### For Systems with GPU:
```bash
# Original model
python -m training.train_lip_reader --batch_size 1 --frames_per_clip 30 --device cuda --num_workers 0

# Lightweight model
python -m training.train_lightweight_lip_reader --batch_size 2 --frames_per_clip 40 --device cuda --num_workers 0
```

## Additional Memory Optimization Tips

### 1. System-Level Optimizations
- Close other applications to free up RAM
- Use CPU-only training if GPU memory is insufficient
- Monitor memory usage with Task Manager (Windows) or htop (Linux)

### 2. Data Loading Optimizations
- Set `--num_workers 0` to reduce memory overhead
- Disable `pin_memory` in DataLoader
- Use smaller frame sequences

### 3. Model Optimizations
- Enable gradient checkpointing (already implemented)
- Use mixed precision training (if supported)
- Reduce model complexity (use lightweight model)

### 4. Training Optimizations
- Use gradient accumulation for effective larger batch sizes
- Implement early stopping to reduce training time
- Save checkpoints less frequently

## Expected Memory Usage

| Model | Batch Size | Frames | Estimated RAM |
|-------|------------|--------|---------------|
| Original | 1 | 30 | ~2-3 GB |
| Original | 2 | 30 | ~4-6 GB |
| Lightweight | 1 | 40 | ~1-2 GB |
| Lightweight | 2 | 40 | ~2-3 GB |

## Troubleshooting

### If you still get memory errors:
1. **Reduce batch size further**: Try `--batch_size 1`
2. **Reduce frames per clip**: Try `--frames_per_clip 20`
3. **Use CPU only**: Add `--device cpu`
4. **Use lightweight model**: Switch to `train_lightweight_lip_reader.py`
5. **Reduce speakers**: Train with fewer speakers initially

### If training is too slow:
1. **Increase batch size** if memory allows
2. **Use GPU** if available
3. **Reduce validation frequency**
4. **Use fewer epochs** for initial testing

## Success Indicators

When the training starts successfully, you should see:
```
Using device: cpu
Train samples: [number]
Val samples: [number]
Epoch 1/10
Train:   0%|          | 0/[number] [00:00<?, ?it/s]
```

The training should proceed without memory errors and show loss values for each epoch.

## Next Steps

1. Start with the lightweight model and small parameters
2. Gradually increase batch size and frames if memory allows
3. Once stable, experiment with the original model
4. Consider using gradient accumulation for better training stability

Remember: It's better to train successfully with smaller parameters than to fail with larger ones. You can always scale up once you have a working baseline. 