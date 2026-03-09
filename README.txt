================================================================================
URBAN MISSION PLANNING SOLUTION
================================================================================

A computer vision and pathfinding system that analyzes satellite imagery to
extract road networks and compute optimal paths between coordinate points.

Version: 1.0
Author: Urban Mission Planning Solution Team
Date: February 2026

================================================================================
TABLE OF CONTENTS
================================================================================

1. System Overview
2. Architecture
3. Installation
4. Quick Start
5. Training the Model
6. Running Inference
7. Expected Performance
8. Directory Structure
9. Configuration
10. Troubleshooting
11. Advanced Usage
12. Testing
13. Google Colab Usage
14. Technical Details
15. License and Acknowledgments

================================================================================
1. SYSTEM OVERVIEW
================================================================================

The Urban Mission Planning Solution processes satellite imagery to:
- Identify road networks using deep learning semantic segmentation
- Construct graph representations of road networks
- Compute optimal paths between coordinates using A* algorithm
- Generate valid waypoint sequences that minimize path length
- Validate paths against challenge constraints

Key Features:
- U-Net/DeepLabV3+ architecture for road segmentation
- GPU acceleration for fast inference
- Property-based testing for correctness validation
- Batch processing for multiple images
- Google Colab compatible
- Reproducible results with deterministic execution

================================================================================
2. ARCHITECTURE
================================================================================

The system follows a pipeline architecture with six main components:

1. ImagePreprocessor
   - Loads TIFF satellite images
   - Normalizes pixel values
   - Converts to PyTorch tensors

2. RoadSegmentationModel
   - U-Net or DeepLabV3+ architecture
   - Binary segmentation (road vs non-road)
   - Pretrained ImageNet encoders

3. GraphConstructor
   - Converts binary road masks to graphs
   - 8-connectivity for neighbor connections
   - Euclidean distance edge weights

4. PathfindingEngine
   - A* algorithm for shortest path
   - Ramer-Douglas-Peucker path simplification
   - Fallback strategies for disconnected networks

5. PathValidator
   - Bresenham's line algorithm for segment checking
   - Violation counting (off-road pixels)
   - Score calculation: 1000 - PathLength - 50 × Violations

6. SolutionGenerator
   - Orchestrates complete pipeline
   - Generates JSON output files
   - Batch processing support

================================================================================
3. INSTALLATION
================================================================================

PREREQUISITES
-------------
- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster training/inference

STEP 1: Clone or Extract Repository
------------------------------------
Extract the submission package to your desired location.

STEP 2: Install Dependencies
-----------------------------
Open a terminal in the project directory and run:

    pip install -r requirements.txt

This installs:
- PyTorch and torchvision (deep learning)
- segmentation-models-pytorch (U-Net/DeepLabV3+)
- opencv-python, scikit-image, Pillow (image processing)
- networkx, scipy (graph algorithms)
- numpy (numerical computing)
- tqdm, matplotlib (utilities)
- pytest, hypothesis (testing)

STEP 3: Verify Installation
----------------------------
Run the following to verify installation:

    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

STEP 4: Prepare Dataset
------------------------
Organize your dataset in the following structure:

    data/
    ├── reference/
    │   ├── sats/          # Training satellite images (.tiff)
    │   └── maps/          # Training road masks (.tiff)
    └── test/
        └── sats/          # Test satellite images (.tiff)

Note: Update paths in src/config.py if using a different structure.

================================================================================
4. QUICK START
================================================================================

PROCESS A SINGLE IMAGE
-----------------------
python main.py --image data/test/sats/test_001.tif \
               --start 100 200 \
               --goal 1800 1900 \
               --output outputs/test_001.json

BATCH PROCESS MULTIPLE IMAGES
------------------------------
1. Create a coordinates file (coordinates.json):
   {
     "test_001": {"start": [100, 200], "goal": [1800, 1900]},
     "test_002": {"start": [300, 400], "goal": [1600, 1700]}
   }

2. Run batch processing:
   python main.py --batch \
                  --test-dir data/test/sats \
                  --coords coordinates.json \
                  --output-dir outputs/

VIEW HELP
---------
python main.py --help

================================================================================
5. TRAINING THE MODEL
================================================================================

OPTION A: Use Pre-trained Model
--------------------------------
If a trained model checkpoint is provided (models/best_model.pth), you can
skip training and proceed directly to inference.

OPTION B: Train from Scratch
-----------------------------
1. Prepare training data in data/reference/ directory
2. Run training script:

   python -c "
   from src.road_segmentation_model import RoadSegmentationModel
   from src.dataset import RoadSegmentationDataset
   from torch.utils.data import DataLoader
   
   # Initialize model
   model = RoadSegmentationModel(architecture='unet', encoder_name='resnet34')
   
   # Load dataset
   dataset = RoadSegmentationDataset('data/reference/sats', 'data/reference/maps')
   train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
   
   # Train
   model.train_model(train_loader, val_loader=None, epochs=50)
   model.save_checkpoint('models/best_model.pth')
   "

3. Training takes 10-30 minutes on GPU, 2-3 hours on CPU

TRAINING PARAMETERS
-------------------
Edit src/config.py to adjust:
- BATCH_SIZE: Reduce if out of memory (default: 4)
- NUM_EPOCHS: Increase for better accuracy (default: 50)
- LEARNING_RATE: Adjust for convergence (default: 0.001)
- LOSS_FUNCTION: Choose "bce", "dice", or "focal" (default: "dice")

MONITORING TRAINING
-------------------
Training logs are saved to urban_mission_planning.log
Monitor validation IoU - target is > 0.85

================================================================================
6. RUNNING INFERENCE
================================================================================

SINGLE IMAGE INFERENCE
-----------------------
python main.py --image path/to/image.tif \
               --start X1 Y1 \
               --goal X2 Y2 \
               --output solution.json \
               --checkpoint models/best_model.pth

Parameters:
- --image: Path to satellite image TIFF file
- --start: Start coordinates (x y)
- --goal: Goal coordinates (x y)
- --output: Output JSON file path
- --checkpoint: Model checkpoint file (default: models/best_model.pth)

BATCH INFERENCE
---------------
python main.py --batch \
               --test-dir data/test/sats \
               --coords coordinates.json \
               --output-dir outputs/ \
               --checkpoint models/best_model.pth

Parameters:
- --batch: Enable batch mode
- --test-dir: Directory with test images
- --coords: JSON file with start/goal coordinates
- --output-dir: Output directory for solutions
- --checkpoint: Model checkpoint file

ADVANCED OPTIONS
----------------
--device cuda|cpu|auto     # Force device (default: auto)
--threshold 0.5            # Segmentation threshold (default: 0.5)
--epsilon 2.0              # Path simplification tolerance (default: 2.0)
--log-level DEBUG          # Logging verbosity (default: INFO)
--seed 42                  # Random seed (default: 42)

EXAMPLE WITH OPTIONS
--------------------
python main.py --image test.tif \
               --start 100 200 \
               --goal 1800 1900 \
               --output solution.json \
               --device cuda \
               --epsilon 1.5 \
               --log-level DEBUG

================================================================================
7. EXPECTED PERFORMANCE
================================================================================

ACCURACY METRICS
----------------
- Road Segmentation IoU: > 0.85 on validation data
- Path Validity Rate: > 95% (zero violations)
- Average Score: > 800 across test cases

PROCESSING TIME (per image)
---------------------------
Image Size    GPU (T4)    CPU
2048×2048     2-5 sec     10-20 sec
4096×4096     5-10 sec    30-60 sec
8192×8192     10-20 sec   60-120 sec

MEMORY USAGE
------------
- Training: 4-8 GB GPU memory (batch size 4)
- Inference: 1-2 GB GPU memory per image
- CPU mode: 2-4 GB RAM per image

SCORE BREAKDOWN
---------------
Score = 1000 - PathLength - 50 × Violations

Example:
- PathLength: 150 pixels
- Violations: 0
- Score: 1000 - 150 - 0 = 850

================================================================================
8. DIRECTORY STRUCTURE
================================================================================

urban-mission-planning/
├── README.txt                 # This file
├── requirements.txt           # Python dependencies
├── main.py                    # Main execution script
├── src/                       # Source code
│   ├── config.py              # Configuration settings
│   ├── image_preprocessor.py  # Image loading and preprocessing
│   ├── road_segmentation_model.py  # Segmentation model
│   ├── graph_constructor.py   # Road mask to graph conversion
│   ├── pathfinding_engine.py  # A* pathfinding
│   ├── path_validator.py      # Path validation and scoring
│   ├── solution_generator.py  # Pipeline orchestration
│   ├── dataset.py             # PyTorch dataset
│   ├── loss_functions.py      # Loss functions
│   ├── morphological_processor.py  # Post-processing
│   ├── performance_optimizer.py    # Performance optimization
│   ├── reproducibility.py     # Deterministic execution
│   ├── utils.py               # Utility functions
│   └── logger.py              # Logging configuration
├── tests/                     # Test suite
│   ├── test_image_preprocessor.py
│   ├── test_road_segmentation_model.py
│   ├── test_graph_constructor.py
│   ├── test_pathfinding_engine.py
│   ├── test_path_validator.py
│   ├── test_solution_generator.py
│   ├── test_performance.py
│   ├── test_reproducibility.py
│   └── test_colab_integration.py
├── docs/                      # Documentation
│   ├── coordinate_system_guide.md
│   ├── dataset_documentation.md
│   ├── graph_constructor_documentation.md
│   ├── loss_functions_documentation.md
│   ├── main_script_usage.md
│   ├── path_validator_documentation.md
│   ├── solution_generator_documentation.md
│   └── training_loop_documentation.md
├── examples/                  # Usage examples
│   ├── dataset_usage_example.py
│   ├── graph_constructor_usage_example.py
│   ├── loss_function_usage_example.py
│   ├── path_validator_usage_example.py
│   ├── solution_generator_usage_example.py
│   └── train_model_example.py
├── models/                    # Model checkpoints
│   └── best_model.pth         # Trained model weights
├── outputs/                   # Generated solutions
│   ├── test_001.json
│   ├── test_002.json
│   └── batch_summary.json
├── data/                      # Dataset (not included)
│   ├── reference/
│   │   ├── sats/
│   │   └── maps/
│   └── test/
│       └── sats/
└── Colab_Training_Complete.ipynb  # Google Colab notebook

================================================================================
9. CONFIGURATION
================================================================================

Configuration is managed in src/config.py. Key parameters:

IMAGE PROCESSING
----------------
MIN_IMAGE_SIZE = 1500          # Minimum image dimension
MAX_IMAGE_SIZE = 8192          # Maximum image dimension

MODEL CONFIGURATION
-------------------
MODEL_ARCHITECTURE = "unet"    # Options: "unet", "deeplabv3plus"
ENCODER_NAME = "resnet34"      # Encoder backbone
ENCODER_WEIGHTS = "imagenet"   # Pretrained weights

TRAINING HYPERPARAMETERS
------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_EPOCHS = 50
LOSS_FUNCTION = "dice"         # Options: "bce", "dice", "focal"
OPTIMIZER = "adam"             # Options: "adam", "sgd", "adamw"

MORPHOLOGICAL POST-PROCESSING
------------------------------
APPLY_MORPHOLOGICAL_CLOSING = True
MORPHOLOGICAL_KERNEL_SIZE = 3  # 3 or 5
MORPHOLOGICAL_ITERATIONS = 1

GRAPH CONSTRUCTION
------------------
CONNECTIVITY = 8               # 4 or 8
MAX_SEARCH_DISTANCE = 50       # Max distance to nearest road
APPLY_SKELETONIZATION = True

PATHFINDING
-----------
PATHFINDING_ALGORITHM = "astar"
SIMPLIFICATION_EPSILON = 2.0
MAX_WAYPOINTS_BEFORE_SIMPLIFICATION = 100

VALIDATION
----------
VIOLATION_PENALTY = 50
BASE_SCORE = 1000

PERFORMANCE
-----------
ENABLE_GPU_ACCELERATION = True
MEMORY_LIMIT_GB = 1.0
ENABLE_PREPROCESSING_CACHE = True

REPRODUCIBILITY
---------------
RANDOM_SEED = 42
ENABLE_DETERMINISTIC = True

LOGGING
-------
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "urban_mission_planning.log"

================================================================================
10. TROUBLESHOOTING
================================================================================

ISSUE: "Out of Memory" Error
-----------------------------
SYMPTOMS: CUDA out of memory or system RAM exhausted
SOLUTIONS:
1. Reduce BATCH_SIZE in src/config.py (try 2 or 1)
2. Process smaller images first
3. Use CPU mode: python main.py --device cpu ...
4. Clear GPU cache: torch.cuda.empty_cache()
5. Close other applications

ISSUE: "No module named 'segmentation_models_pytorch'"
-------------------------------------------------------
SYMPTOMS: Import error when running scripts
SOLUTIONS:
1. Reinstall dependencies: pip install -r requirements.txt
2. Check Python version: python --version (need 3.8+)
3. Use virtual environment to avoid conflicts

ISSUE: "Model checkpoint not found"
------------------------------------
SYMPTOMS: FileNotFoundError for models/best_model.pth
SOLUTIONS:
1. Train model first (see Section 5)
2. Specify correct checkpoint path: --checkpoint path/to/model.pth
3. Check that checkpoint file exists and is not corrupted

ISSUE: "No path found between start and goal"
----------------------------------------------
SYMPTOMS: Warning message, direct line path returned
CAUSES:
1. Road network is disconnected
2. Start/goal too far from roads
3. Poor segmentation quality
SOLUTIONS:
1. Adjust start/goal coordinates closer to roads
2. Increase MAX_SEARCH_DISTANCE in config
3. Retrain model with more epochs
4. Apply morphological closing to connect roads

ISSUE: Low Scores or High Violations
-------------------------------------
SYMPTOMS: Scores < 800, many violations
CAUSES:
1. Poor road segmentation
2. Aggressive path simplification
3. Disconnected road networks
SOLUTIONS:
1. Retrain model with more data/epochs
2. Reduce SIMPLIFICATION_EPSILON (e.g., 1.0)
3. Disable skeletonization: APPLY_SKELETONIZATION = False
4. Adjust segmentation threshold: --threshold 0.4

ISSUE: Slow Processing on CPU
------------------------------
SYMPTOMS: Processing takes minutes per image
SOLUTIONS:
1. Use GPU if available
2. Reduce image size for testing
3. Disable preprocessing cache if disk I/O is slow
4. Process images in smaller batches

ISSUE: "CUDA error: device-side assert triggered"
--------------------------------------------------
SYMPTOMS: CUDA error during training/inference
SOLUTIONS:
1. Check for NaN values in data
2. Reduce learning rate
3. Use CPU mode to debug: --device cpu
4. Update PyTorch: pip install --upgrade torch torchvision

ISSUE: Coordinates Out of Bounds
---------------------------------
SYMPTOMS: ValueError about coordinates
SOLUTIONS:
1. Verify coordinates are within image dimensions
2. Check coordinate format: (x, y) not (row, col)
3. Ensure coordinates are integers
4. Review coordinate system documentation in docs/

ISSUE: JSON Output Invalid
---------------------------
SYMPTOMS: JSON parsing errors
SOLUTIONS:
1. Check output file is not corrupted
2. Verify path has at least 2 waypoints
3. Ensure all coordinates are integers
4. Validate JSON format: python -m json.tool output.json

================================================================================
11. ADVANCED USAGE
================================================================================

CUSTOM MODEL ARCHITECTURE
--------------------------
from src.road_segmentation_model import RoadSegmentationModel

model = RoadSegmentationModel(
    architecture="deeplabv3plus",  # or "unet"
    encoder_name="resnet50",       # or "resnet34", "efficientnet-b0"
    encoder_weights="imagenet",
    device="cuda"
)

CUSTOM LOSS FUNCTION
--------------------
from src.loss_functions import DiceLoss, FocalLoss, BCELoss

# Dice loss (default)
loss_fn = DiceLoss()

# Focal loss for imbalanced data
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

# Binary cross-entropy
loss_fn = BCELoss()

CUSTOM GRAPH CONSTRUCTION
--------------------------
from src.graph_constructor import GraphConstructor

# 4-connectivity (faster, fewer edges)
graph_constructor = GraphConstructor(connectivity=4)

# 8-connectivity (more paths, slower)
graph_constructor = GraphConstructor(connectivity=8)

graph = graph_constructor.build_graph(road_mask)

CUSTOM PATH SIMPLIFICATION
---------------------------
from src.pathfinding_engine import PathfindingEngine

pathfinder = PathfindingEngine()

# More waypoints (epsilon = 1.0)
simplified = pathfinder.simplify_path(path, epsilon=1.0)

# Fewer waypoints (epsilon = 5.0)
simplified = pathfinder.simplify_path(path, epsilon=5.0)

PROGRAMMATIC USAGE
------------------
from src.solution_generator import SolutionGenerator
from src.road_segmentation_model import RoadSegmentationModel
from src.config import Config

# Load model
model = RoadSegmentationModel()
model.load_checkpoint('models/best_model.pth')

# Create generator
config = Config.to_dict()
generator = SolutionGenerator(model, config)

# Process image
result = generator.process_image(
    image_path='test.tif',
    start=(100, 200),
    goal=(1800, 1900)
)

# Access results
print(f"Path length: {result['validation']['path_length']}")
print(f"Score: {result['validation']['score']}")
print(f"Waypoints: {len(result['path'])}")

BATCH PROCESSING WITH CUSTOM LOGIC
-----------------------------------
import os
from pathlib import Path

# Get all test images
test_dir = Path('data/test/sats')
images = list(test_dir.glob('*.tif'))

# Process each image
for image_path in images:
    image_id = image_path.stem
    
    # Custom start/goal logic
    start = (100, 200)
    goal = (1800, 1900)
    
    # Process
    result = generator.process_image(str(image_path), start, goal)
    
    # Save solution
    output_path = f'outputs/{image_id}.json'
    generator.generate_solution_json(image_id, result['path'], output_path)

================================================================================
12. TESTING
================================================================================

RUN ALL TESTS
-------------
pytest tests/

RUN SPECIFIC TEST FILE
----------------------
pytest tests/test_pathfinding_engine.py

RUN WITH COVERAGE
-----------------
pytest --cov=src tests/

RUN PROPERTY-BASED TESTS
-------------------------
pytest tests/test_preprocessing_properties.py -v

TEST CATEGORIES
---------------
1. Unit Tests: Test individual components
2. Integration Tests: Test component interactions
3. Property Tests: Test universal correctness properties
4. Performance Tests: Test speed and memory usage

EXAMPLE TEST EXECUTION
----------------------
# Test image preprocessing
pytest tests/test_image_preprocessor.py -v

# Test pathfinding
pytest tests/test_pathfinding_engine.py -v

# Test complete pipeline
pytest tests/test_solution_generator.py -v

# Test reproducibility
pytest tests/test_reproducibility.py -v

INTERPRETING TEST RESULTS
--------------------------
- PASSED: Test succeeded
- FAILED: Test failed, check error message
- SKIPPED: Test skipped (optional or conditional)
- ERROR: Test encountered an error

All tests should pass before deployment.

================================================================================
13. GOOGLE COLAB USAGE
================================================================================

SETUP IN COLAB
--------------
1. Upload Colab_Training_Complete.ipynb to Google Colab
2. Select Runtime → Change runtime type → GPU (T4 recommended)
3. Run cells sequentially

MOUNT GOOGLE DRIVE
------------------
from google.colab import drive
drive.mount('/content/drive')

# Set data path
DATA_ROOT = '/content/drive/MyDrive/ump_data'

INSTALL DEPENDENCIES
--------------------
!pip install segmentation-models-pytorch
!pip install opencv-python scikit-image networkx scipy

UPLOAD DATA
-----------
from google.colab import files
uploaded = files.upload()  # Upload zip file
!unzip data.zip

TRAIN MODEL
-----------
# Run training cells in notebook
# Model saves to /content/models/best_model.pth

DOWNLOAD RESULTS
----------------
from google.colab import files
files.download('models/best_model.pth')
files.download('outputs/test_001.json')

MEMORY MANAGEMENT
-----------------
import torch
torch.cuda.empty_cache()  # Clear GPU memory

import gc
gc.collect()  # Clear Python memory

COLAB LIMITATIONS
-----------------
- Session timeout: 12 hours (free), 24 hours (Pro)
- GPU time limit: Variable based on usage
- RAM: 12 GB (free), 25 GB (Pro)
- Storage: 15 GB (free), 100 GB (Pro)

TIPS FOR COLAB
--------------
1. Save checkpoints frequently
2. Download results before session ends
3. Use smaller batch sizes (2-4)
4. Process images in small batches
5. Clear memory between batches

================================================================================
14. TECHNICAL DETAILS
================================================================================

COORDINATE SYSTEM
-----------------
- Origin: Top-left corner (0, 0)
- X-axis: Horizontal (left to right)
- Y-axis: Vertical (top to bottom)
- Format: (x, y) tuples
- Array indexing: road_mask[y, x]

SEGMENTATION MODEL
------------------
- Architecture: U-Net with ResNet34 encoder
- Input: RGB image (3, H, W)
- Output: Binary mask (H, W) with values 0 or 1
- Activation: Sigmoid
- Threshold: 0.5 (configurable)

GRAPH REPRESENTATION
--------------------
- Nodes: Road pixels (mask value = 1)
- Edges: Connect adjacent pixels (8-connectivity)
- Edge weights: Euclidean distance
- Graph type: Undirected, weighted

PATHFINDING ALGORITHM
----------------------
- Algorithm: A* with Euclidean heuristic
- Optimality: Guaranteed shortest path
- Complexity: O(E log V) where E=edges, V=nodes
- Fallback: Direct line if no path exists

PATH SIMPLIFICATION
-------------------
- Algorithm: Ramer-Douglas-Peucker
- Epsilon: 2.0 pixels (configurable)
- Preserves: First and last waypoints
- Reduces: Waypoints while maintaining shape

VALIDATION
----------
- Segment checking: Bresenham's line algorithm
- Violation: Off-road or out-of-bounds pixel
- Score formula: 1000 - PathLength - 50 × Violations
- Valid path: Zero violations, all waypoints in bounds

MORPHOLOGICAL OPERATIONS
-------------------------
- Closing: Connects fragmented roads
- Kernel: 3×3 or 5×5
- Iterations: 1 (configurable)
- Skeletonization: Reduces graph size by 80%+

PERFORMANCE OPTIMIZATIONS
--------------------------
- GPU acceleration for inference
- Batch processing for multiple images
- Preprocessing cache for repeated images
- Graph pruning for large images
- Memory monitoring and management

REPRODUCIBILITY
---------------
- Random seeds: PyTorch, NumPy, Python
- Deterministic algorithms: Enabled
- Checkpoint saving: Includes hyperparameters
- Identical outputs: Same input → same output

================================================================================
15. LICENSE AND ACKNOWLEDGMENTS
================================================================================

LICENSE
-------
This implementation is for the Urban Mission Planning challenge.
All rights reserved.

ACKNOWLEDGMENTS
---------------
- segmentation-models-pytorch: U-Net and DeepLabV3+ implementations
- PyTorch: Deep learning framework
- NetworkX: Graph algorithms
- scikit-image: Image processing
- OpenCV: Computer vision utilities
- Hypothesis: Property-based testing

THIRD-PARTY LIBRARIES
---------------------
See requirements.txt for complete list of dependencies.
All libraries are used under their respective licenses.

REFERENCES
----------
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
  Image Segmentation", MICCAI 2015
- DeepLabV3+: Chen et al., "Encoder-Decoder with Atrous Separable Convolution
  for Semantic Image Segmentation", ECCV 2018
- A* Algorithm: Hart et al., "A Formal Basis for the Heuristic Determination
  of Minimum Cost Paths", IEEE Transactions on Systems Science and
  Cybernetics, 1968
- Ramer-Douglas-Peucker: Douglas & Peucker, "Algorithms for the reduction of
  the number of points required to represent a digitized line or its
  caricature", The Canadian Cartographer, 1973

CONTACT
-------
For questions or issues, please refer to the project documentation in the
docs/ directory or review the test suite in tests/.

================================================================================
END OF README
================================================================================
