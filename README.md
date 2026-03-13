# UND Map - Geospatial AI Dashboard

A modern, full-stack application for analyzing satellite imagery using advanced AI algorithms with interactive 3D visualization and real-time processing analytics.

## Overview

The application combines a powerful Python backend for geospatial analysis with a modern Next.js frontend featuring:
- Interactive satellite imagery analysis
- AI-powered road segmentation and pathfinding
- Graph construction from satellite images
- Optimal route calculation using A* algorithm
- Real-time visualization and performance monitoring
- 3D network visualization of the processing pipeline

## Quick Start

### Prerequisites

- Node.js 18+ and npm/pnpm
- Python 3.8+
- Git

### Option 1: Full Setup (Frontend + Backend)

1. Clone the repository:
   ```bash
   git clone https://github.com/Tanishqyadav937/und_map.git
   cd und_map
   ```

2. Install dependencies:
   ```bash
   npm install
   pip install -r requirements.txt
   pip install -r requirements-flask.txt
   ```

3. Start both services:
   - Terminal 1 - Backend:
     ```bash
     python flask_app.py
     ```
   - Terminal 2 - Frontend:
     ```bash
     npm run dev
     ```

4. Open http://localhost:3000 in your browser

### Option 2: Frontend Only (UI Development)

```bash
npm install
npm run dev
```

## Key Features

### Frontend Dashboard
- **Interactive Satellite Map**: Canvas-based 2D visualization with zoom, pan, and grid overlay
- **Real-time Statistics**: System overview with processing metrics
- **Processing Controls**: Image upload and batch processing interface
- **3D Network Visualization**: Interactive Three.js visualization of the processing pipeline
- **Performance Analytics**: Charts tracking processing time, accuracy, and coverage
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Glassmorphism UI**: Modern dark theme with neon cyan and purple accents

### Backend API
- **Image Upload**: RESTful endpoint for satellite image uploads
- **Processing Pipeline**: Integration with Python geospatial analysis engine
- **Batch Processing**: Support for processing multiple images simultaneously
- **Real-time Status**: Health checks and system status monitoring
- **Error Handling**: Comprehensive error reporting and logging

### Processing Pipeline
- **Road Segmentation**: U-Net based deep learning model for road detection
- **Graph Construction**: Converts segmented roads to navigation graphs
- **Pathfinding**: A* algorithm for optimal route calculation
- **Path Optimization**: Simplification and smoothing of generated paths
- **Validation**: Ensures paths stay on valid road networks

## System Requirements

### Development Environment
- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 5 GB for project and dependencies
- **GPU**: Optional (NVIDIA with CUDA for faster processing)

### Deployment Requirements
- **Server**: Any Linux/Windows server with Python and Node.js
- **Port Access**: 3000 (frontend), 8000 (backend)
- **Environment Variables**: See `.env.example`

## Expected Performance

### Processing Time (per satellite image)
- **High-end GPU**: 2-5 seconds
- **Mid-range GPU**: 5-15 seconds
- **CPU Only**: 10-30 seconds

### Model Accuracy
- **Road Segmentation IoU**: > 0.85
- **Path Validity**: > 95%
- **Average Processing Score**: > 800

### UI Performance
- **Dashboard Load Time**: < 2 seconds
- **Map Zoom/Pan**: 60 FPS
- **3D Visualization**: 30-60 FPS

## Troubleshooting

### API Connection Issues

**Problem**: Frontend cannot connect to backend
- Ensure Flask server is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify CORS is enabled in Flask app
- Check firewall settings

### Image Upload Issues

**Problem**: File upload fails
- Maximum file size: 50MB
- Supported formats: PNG, JPG, JPEG, TIFF, BMP, GIF
- Ensure `/uploads` directory has write permissions

### Processing Errors

**Problem**: Image processing fails
- Check Python dependencies: `pip install -r requirements.txt`
- Verify image format and integrity
- Check disk space for temporary files
- Review server logs for detailed errors

### 3D Visualization Not Loading

**Problem**: 3D network visualization doesn't render
- Check browser WebGL support (chrome://gpu in Chrome)
- Verify GPU acceleration is enabled
- Try using a different browser (Chrome/Firefox recommended)
- Check browser console for WebGL errors

### Memory Issues

**Problem**: Out of memory errors during batch processing
- Reduce batch size in configuration
- Process fewer images at once
- Increase available system RAM
- Restart the Flask server between large batches

## Customization

### UI Theme Customization

Edit `tailwind.config.ts` to customize colors:
```typescript
colors: {
  primary: '#00d4ff',      // Cyan
  'primary-dark': '#0099cc',
  accent: '#a855f7',       // Purple
  // ... other colors
}
```

### Backend Configuration

Edit `.env` or create `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
FLASK_ENV=development
PORT=8000
UPLOAD_FOLDER=./uploads
```

### Processing Pipeline Parameters

Modify `src/config.py` for the Python backend:
```python
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
CONNECTIVITY = 4  # Graph connectivity
```

## API Response Format

### Processing Success Response

```json
{
  "success": true,
  "message": "Image processed successfully",
  "data": {
    "graph_path": "/path/to/graph.json",
    "solution": {
      "id": "image_001",
      "path": [[x1, y1], [x2, y2], ...]
    },
    "stats": {
      "processing_time": 4.5,
      "accuracy": 0.92
    }
  }
}
```

### Error Response

```json
{
  "success": false,
  "message": "Detailed error description"
}
```

## Deployment

### Docker Deployment

Build and run using Docker:
```dockerfile
FROM node:18-alpine AS frontend
WORKDIR /app
COPY . .
RUN npm install && npm run build

FROM python:3.10-slim AS backend
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r requirements-flask.txt
EXPOSE 8000 3000
CMD ["python", "flask_app.py"]
```

### Vercel Deployment

1. Connect GitHub repository to Vercel
2. Set environment variables in project settings
3. Deploy frontend as Next.js project
4. Deploy backend as serverless function or separate service

### Self-Hosted Deployment

1. Deploy to Ubuntu/Debian server
2. Install dependencies and configure environment
3. Use systemd to manage services
4. Set up nginx as reverse proxy

## Support & Resources

### Documentation
- API documentation available at `/api/docs`
- Component storybook: `npm run storybook`
- Architecture guide: See `docs/ARCHITECTURE.md`

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Pull Requests: Contribute improvements

### Getting Help
- Check existing issues before creating new ones
- Include error logs and reproduction steps
- Specify your OS, Node/Python version, and browser

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [Three.js](https://threejs.org/) - 3D graphics
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/) - React 3D renderer
- [TailwindCSS](https://tailwindcss.com/) - Styling
- [Recharts](https://recharts.org/) - Charts
- [PyTorch](https://pytorch.org/) - Deep learning
- [NetworkX](https://networkx.org/) - Graph algorithms
- [Lucide Icons](https://lucide.dev/) - Icons

---

**Latest Update**: March 13, 2026
**Version**: 1.0.0
**Status**: Active Development
