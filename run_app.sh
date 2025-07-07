#!/bin/bash

# Stock Prediction Dashboard Launcher
echo "🚀 Starting Stock Prediction Dashboard..."

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 found"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, yfinance, pandas, numpy, plotly, pandas_ta" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies are installed"
else
    echo "⚠️  Some dependencies missing. Installing..."
    pip3 install -r requirements.txt
fi

# Launch the app
echo "🌐 Launching dashboard..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python3 -m streamlit run app.py 