#!/bin/bash
# Quick setup and run script for PDF processing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Homicide Review PDF Processing Setup"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

echo "Python: $(python3 --version)"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Run script
echo ""
echo "Starting PDF scraping and extraction..."
echo ""

python3 scrape_and_extract.py

echo ""
echo "=========================================="
echo "Processing complete!"
echo "=========================================="
