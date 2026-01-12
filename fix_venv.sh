#!/bin/bash
# Fix virtual environment path issues

echo "Recreating virtual environment..."
rm -rf myenv
python3 -m venv myenv

echo "Activating virtual environment..."
source myenv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Done! Virtual environment fixed."
echo "To use it: source myenv/bin/activate"
