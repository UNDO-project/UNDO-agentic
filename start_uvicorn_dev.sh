#!/bin/bash
# Start FastAPI server
# Note: Server binds to 0.0.0.0 but you should access it via:
#   - http://localhost:8080/docs
#   - http://127.0.0.1:8080/docs

echo "Starting FastAPI server..."
echo "Access the API at:"
echo "  - Swagger UI: http://localhost:8080/docs"
echo "  - ReDoc:      http://localhost:8080/redoc"
echo "  - Health:     http://localhost:8080/health"
echo ""

uvicorn src.api.main:app --port=8080 --host='0.0.0.0' --reload --loop asyncio