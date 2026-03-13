"""
Flask REST API wrapper for the UND Map geospatial analysis pipeline.
This server provides endpoints for the Next.js frontend to communicate
with the Python image processing and graph analysis backend.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import traceback
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the solution generator and config
try:
    from solution_generator import SolutionGenerator
    from config import UPLOAD_FOLDER
except ImportError as e:
    print(f"Error importing modules: {e}")
    UPLOAD_FOLDER = './uploads'

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize solution generator
try:
    solution_generator = SolutionGenerator()
except Exception as e:
    print(f"Warning: Could not initialize SolutionGenerator: {e}")
    solution_generator = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'gif'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def log_request(method, endpoint, status=None, message=None):
    """Log API requests for debugging."""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {method} {endpoint}"
    if status:
        log_entry += f" - Status: {status}"
    if message:
        log_entry += f" - Message: {message}"
    print(log_entry)


# Health check endpoint
@app.route('/status', methods=['GET'])
def status():
    """Check API status and backend availability."""
    log_request('GET', '/status')
    
    try:
        backend_available = solution_generator is not None
        return jsonify({
            'status': 'healthy' if backend_available else 'degraded',
            'message': 'API is running' if backend_available else 'Backend not initialized',
            'timestamp': datetime.now().isoformat(),
            'backend': 'available' if backend_available else 'unavailable'
        }), 200
    except Exception as e:
        log_request('GET', '/status', 500, str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload."""
    log_request('POST', '/upload')
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            log_request('POST', '/upload', 400, 'No file provided')
            return jsonify({
                'success': False,
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Check if file has filename
        if file.filename == '':
            log_request('POST', '/upload', 400, 'Empty filename')
            return jsonify({
                'success': False,
                'message': 'Empty filename'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            log_request('POST', '/upload', 400, 'File type not allowed')
            return jsonify({
                'success': False,
                'message': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        log_request('POST', '/upload', 200, f'File saved: {filename}')
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'file_path': filepath,
            'file_name': filename
        }), 200
        
    except Exception as e:
        log_request('POST', '/upload', 500, str(e))
        return jsonify({
            'success': False,
            'message': f'Upload error: {str(e)}'
        }), 500


# Process image endpoint
@app.route('/process', methods=['POST'])
def process_image():
    """Process satellite image and generate solution."""
    log_request('POST', '/process')
    
    try:
        if not solution_generator:
            log_request('POST', '/process', 503, 'Backend not initialized')
            return jsonify({
                'success': False,
                'message': 'Backend processing engine not initialized'
            }), 503
        
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            log_request('POST', '/process', 400, 'Missing image_path')
            return jsonify({
                'success': False,
                'message': 'Missing required field: image_path'
            }), 400
        
        image_path = data['image_path']
        
        # Check if file exists
        if not os.path.exists(image_path):
            log_request('POST', '/process', 404, f'Image not found: {image_path}')
            return jsonify({
                'success': False,
                'message': f'Image file not found: {image_path}'
            }), 404
        
        # Process the image
        print(f"Processing image: {image_path}")
        result = solution_generator.process_satellite_image(image_path)
        
        if result and result.get('success'):
            log_request('POST', '/process', 200, 'Processing completed')
            return jsonify({
                'success': True,
                'message': 'Image processed successfully',
                'data': {
                    'graph_path': result.get('graph_path'),
                    'solution': result.get('solution'),
                    'stats': result.get('stats')
                }
            }), 200
        else:
            error_msg = result.get('message') if result else 'Unknown processing error'
            log_request('POST', '/process', 400, error_msg)
            return jsonify({
                'success': False,
                'message': error_msg
            }), 400
            
    except Exception as e:
        log_request('POST', '/process', 500, str(e))
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Processing error: {str(e)}',
            'error_details': traceback.format_exc()
        }), 500


# Batch processing endpoint
@app.route('/batch-process', methods=['POST'])
def batch_process():
    """Process multiple images."""
    log_request('POST', '/batch-process')
    
    try:
        if not solution_generator:
            return jsonify({
                'success': False,
                'message': 'Backend not initialized'
            }), 503
        
        data = request.get_json()
        
        if not data or 'image_paths' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required field: image_paths'
            }), 400
        
        image_paths = data['image_paths']
        
        if not isinstance(image_paths, list):
            return jsonify({
                'success': False,
                'message': 'image_paths must be a list'
            }), 400
        
        results = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                result = solution_generator.process_satellite_image(image_path)
                results.append({
                    'image_path': image_path,
                    'success': result.get('success', False) if result else False,
                    'data': result.get('data') if result else None
                })
            else:
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': 'File not found'
                })
        
        log_request('POST', '/batch-process', 200, f'Processed {len(results)} images')
        return jsonify({
            'success': True,
            'message': f'Batch processing completed',
            'results': results,
            'total': len(results)
        }), 200
        
    except Exception as e:
        log_request('POST', '/batch-process', 500, str(e))
        return jsonify({
            'success': False,
            'message': f'Batch processing error: {str(e)}'
        }), 500


# Health probe endpoint for load balancers
@app.route('/health', methods=['GET'])
def health():
    """Simple health check for load balancers."""
    return jsonify({'status': 'ok'}), 200


# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'name': 'UND Map API',
        'version': '1.0.0',
        'description': 'Geospatial Analysis and Pathfinding REST API',
        'endpoints': {
            'GET /status': 'Check API and backend status',
            'GET /health': 'Health check for load balancers',
            'POST /upload': 'Upload satellite image',
            'POST /process': 'Process satellite image',
            'POST /batch-process': 'Process multiple images'
        }
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'message': 'Method not allowed'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting UND Map API server on port {port}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
