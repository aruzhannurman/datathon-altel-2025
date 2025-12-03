# Instagram Comment Responder

A Streamlit application with FastAPI backend for processing Instagram post comments and generating AI-powered responses.

## Features

- **Process All Comments**: Input an Instagram post URL, scrape all comments, generate AI responses, and download results as XLSX.
- **Process Single Comment**: Input a post URL and a specific comment, generate an AI response.
- Progress bars and animations for a visually appealing experience.

## 🐳 Docker Setup (Recommended)

### Quick Start with Docker Compose

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd datathon-tele2-2025
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the applications**:
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

### Manual Docker Build

**Build and run FastAPI backend**:
```bash
docker build -f Dockerfile.fastapi -t instagram-comment-fastapi .
docker run -p 8000:8000 -v $(pwd)/data:/app/data instagram-comment-fastapi
```

**Build and run Streamlit frontend**:
```bash
docker build -f Dockerfile.streamlit -t instagram-comment-streamlit .
docker run -p 8501:8501 -e FASTAPI_URL=http://localhost:8000 instagram-comment-streamlit
```

## 🛠️ Local Development Setup

### FastAPI Backend

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI backend**:
   ```bash
   python run_server.py
   ```
   Or manually:
   ```bash
   uvicorn main:app --reload
   ```

### Streamlit Frontend

1. **Install dependencies**:
   ```bash
   pip install -r requirements-streamlit.txt
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 🧪 Testing

To test the progress tracking, you can use:
```bash
python test_progress.py
```

## Usage

- For processing all comments: Enter the Instagram post URL and click "Process". Monitor the progress bar.
- For single comment: Enter the URL and comment, then click "Generate Answer".

## Progress Tracking

The application now tracks progress through 4 main steps:
1. Toxicity Detection
2. Spam Detection  
3. Comment Classification
4. Answer Generation

Progress is updated in real-time and displayed in the Streamlit interface.

## 📁 Project Structure

```
├── app.py                      # Streamlit frontend
├── main.py                     # FastAPI backend
├── requirements.txt            # Backend dependencies
├── requirements-streamlit.txt  # Frontend dependencies
├── Dockerfile.fastapi          # FastAPI Docker image
├── Dockerfile.streamlit        # Streamlit Docker image
├── docker-compose.yml          # Orchestration
├── models/                     # AI pipeline modules
├── api/                        # API utilities
├── detects/                    # Detection modules
├── classifiers/                # Classification modules
└── data/                       # Data storage
```

## 🚀 Production Deployment

For production deployment, consider:

1. **Environment Variables**: Set proper environment variables
2. **Secrets Management**: Secure API keys and credentials
3. **Resource Limits**: Configure appropriate CPU/memory limits
4. **Persistent Storage**: Mount volumes for data persistence
5. **Load Balancing**: Use reverse proxy for scaling

## Notes

- Instagram scraping may require login for private posts.
- The AI model uses GPT-4o-mini for response generation.
- Ensure both servers are running for the app to work.
- Check the FastAPI logs for detailed processing information.

## 🐛 Troubleshooting

- **Port conflicts**: Change ports in docker-compose.yml if needed
- **Memory issues**: Increase Docker memory limits for ML models
- **Network issues**: Ensure containers can communicate via the bridge network# datathon-activ-2025
# datathon-altel-2025
