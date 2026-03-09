# Urban Mission Planning Solution - Google Colab Notebook

## Overview

This Google Colab notebook provides a complete demonstration of the Urban Mission Planning Solution, including:
- Road segmentation using deep learning (U-Net)
- Graph construction from road masks
- Optimal pathfinding using A* algorithm
- Path simplification and validation
- Visualization and solution generation

## Quick Start

### Option 1: Open in Google Colab

1. Upload `urban_mission_planning_colab.ipynb` to Google Colab
2. Click **Runtime → Change runtime type** and select **GPU** (recommended)
3. Run all cells sequentially

### Option 2: Direct Link

If the notebook is hosted on GitHub or Google Drive:
- Click the "Open in Colab" badge (if available)
- Or manually upload to Colab

## Prerequisites

### Data Requirements

You need the Urban Mission Planning dataset with the following structure:
```
ump_data/
├── reference/
│   ├── sats/          # Training satellite images (.tiff)
│   └── maps/          # Training road masks (.tiff)
└── test/
    └── sats/          # Test satellite images (.tiff)
```

### Data Setup Options

**Option A: Google Drive (Recommended)**
1. Upload your dataset to Google Drive
2. In the notebook, run the "Mount Google Drive" cell
3. Update the `DATA_ROOT` path to match your Drive structure

**Option B: Direct Upload**
1. Zip your dataset
2. Run the "Upload Data" cell in the notebook
3. Upload the zip file when prompted

## Notebook Structure

### 1. Environment Setup (Cells 1-2)
- Installs all required dependencies
- Detects GPU/CPU and configures PyTorch
- Sets random seeds for reproducibility

### 2. Data Loading (Cells 3-4)
- Mounts Google Drive or handles direct uploads
- Configures data paths

### 3. Source Code Creation (Cells 5-10)
- Creates Python modules for all components:
  - `config.py` - Configuration settings
  - `image_preprocessor.py` - Image loading and preprocessing
  - `road_segmentation_model.py` - U-Net segmentation model
  - `graph_constructor.py` - Road mask to graph conversion
  - `pathfinding_engine.py` - A* pathfinding
  - `dataset.py` - PyTorch dataset
  - `loss_functions.py` - Dice loss

### 4. Model Training (Cell 11)
- Trains U-Net model on reference data
- Uses Dice loss for segmentation
- Saves best model checkpoint
- **Duration**: 10-30 minutes depending on GPU

### 5. Inference Pipeline (Cells 12-13)
- Loads trained model
- Processes test images through complete pipeline
- Generates road masks and paths

### 6. Visualization (Cell 14)
- Displays original image, road mask, and path overlay
- Shows start/goal points and computed path

### 7. Solution Generation (Cell 15)
- Creates JSON output in required format
- Saves solution files

### 8. Memory Management (Cell 16)
- Monitors GPU/CPU memory usage
- Clears cache to prevent OOM errors

### 9. Batch Processing (Cell 17)
- Processes multiple test images
- Generates solutions for all images

### 10. Download Results (Cell 18)
- Downloads trained model
- Downloads solution files

## Hardware Requirements

### Minimum Requirements
- **CPU**: Any modern CPU (training will be slow)
- **RAM**: 4 GB
- **Storage**: 2 GB for code and small dataset

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8+ GB VRAM (T4, P100, V100)
- **RAM**: 12 GB
- **Storage**: 10 GB for full dataset

### Google Colab Tiers
- **Free**: T4 GPU, 12 GB RAM (sufficient for demo)
- **Pro**: Better GPUs, more RAM (recommended for full dataset)
- **Pro+**: Fastest GPUs, most RAM (best for large-scale processing)

## Memory Constraints

Google Colab has memory limits. To stay within constraints:

1. **Reduce Batch Size**: Set `BATCH_SIZE = 1` or `2` in config
2. **Process Fewer Images**: Limit batch processing to 5-10 images
3. **Clear Memory**: Run the memory management cell between batches
4. **Use Smaller Images**: Test on 2048×2048 images first

## Expected Performance

### Training Time (10 epochs)
- **GPU (T4)**: 10-15 minutes
- **CPU**: 2-3 hours

### Inference Time (per image)
- **2048×2048 image**:
  - GPU: 2-5 seconds
  - CPU: 10-20 seconds
- **4096×4096 image**:
  - GPU: 5-10 seconds
  - CPU: 30-60 seconds

### Accuracy Targets
- **Road Segmentation IoU**: > 0.85
- **Path Validity**: > 95% with zero violations
- **Average Score**: > 800

## Troubleshooting

### Issue: "Out of Memory" Error

**Solutions**:
1. Reduce batch size to 1
2. Clear GPU cache: `torch.cuda.empty_cache()`
3. Restart runtime and run again
4. Use smaller images for testing

### Issue: "No module named 'segmentation_models_pytorch'"

**Solution**:
- Re-run the dependency installation cell
- Restart runtime if needed

### Issue: "CUDA out of memory"

**Solutions**:
1. Switch to CPU: Change runtime type to CPU
2. Reduce batch size
3. Process images one at a time

### Issue: "No test images found"

**Solution**:
- Verify `TEST_IMAGES` path is correct
- Check that images have `.tiff` or `.tif` extension
- Ensure data is properly mounted/uploaded

### Issue: "No path found"

**Possible Causes**:
1. Road network is disconnected
2. Start/goal coordinates are too far from roads
3. Poor segmentation quality

**Solutions**:
1. Increase `max_radius` in `add_start_goal_nodes()`
2. Adjust start/goal coordinates
3. Retrain model with more epochs

## Customization

### Change Model Architecture

In the training cell, modify:
```python
model = RoadSegmentationModel(
    architecture="deeplabv3plus",  # or "unet"
    encoder_name="resnet50"        # or "resnet34", "efficientnet-b0"
)
```

### Adjust Training Parameters

In `src/config.py`:
```python
BATCH_SIZE = 2          # Reduce if OOM
NUM_EPOCHS = 20         # Increase for better accuracy
LEARNING_RATE = 0.0001  # Reduce for fine-tuning
```

### Change Pathfinding Settings

```python
# Graph connectivity
graph_constructor = GraphConstructor(connectivity=4)  # or 8

# Path simplification
simplified_path = pathfinder.simplify_path(path, epsilon=1.0)  # Lower = more waypoints
```

### Custom Start/Goal Coordinates

In the inference cell:
```python
# Example: specific coordinates
start = (500, 600)
goal = (2000, 2100)

# Example: random coordinates
import random
start = (random.randint(0, width), random.randint(0, height))
goal = (random.randint(0, width), random.randint(0, height))
```

## Output Format

### Solution JSON Structure

```json
{
  "id": "test_001",
  "path": [
    [100, 200],
    [150, 250],
    [200, 300],
    ...
    [1800, 1900]
  ]
}
```

- `id`: Test image identifier (without extension)
- `path`: List of [x, y] coordinate pairs
- All coordinates are integers
- Minimum 2 waypoints (start and goal)

## Validation

### Path Validation Criteria

1. **Boundary Check**: All waypoints within image bounds
2. **Road Adherence**: All segments stay on road pixels
3. **Connectivity**: Path connects start to goal
4. **Format**: Valid JSON with correct structure

### Score Calculation

```
Score = 1000 - PathLength - 50 × Violations
```

Where:
- `PathLength`: Sum of Euclidean distances between waypoints
- `Violations`: Count of off-road or out-of-bounds pixels

## Additional Resources

### Documentation
- See `docs/` folder for detailed component documentation
- Check `examples/` for usage examples

### Testing
- Run `tests/` to validate implementation
- Use property-based tests for correctness

### Support
- Check GitHub issues for common problems
- Review design document for architecture details

## License

This implementation is for the Urban Mission Planning challenge.

## Acknowledgments

- Uses `segmentation-models-pytorch` for U-Net implementation
- Built with PyTorch, NetworkX, and scikit-image
- Designed for Google Colab environment
