#!/bin/bash
# Quick Start Script for Fashion Product Dataset Generator

echo "=================================="
echo "Fashion Product Dataset Generator"
echo "=================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo " No .env file found!"
    echo ""
    echo "Creating .env from example..."
    cp .env.example .env
    echo ""
    echo "Please edit .env and add your Google Gemini API key:"
    echo "   nano .env"
    echo ""
    echo "Get your free API key from: https://aistudio.google.com/apikey"
    echo ""
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import pandas" 2>/dev/null; then
    echo " Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Run the generator
echo " Running dataset generator..."
echo ""
python3 main.py "$@"
