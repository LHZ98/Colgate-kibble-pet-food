#!/bin/bash
# Quick script to upload Colgate project to GitHub
# Usage: Run this script after creating a GitHub repository

echo "========================================"
echo "Colgate Kibble Segmentation - GitHub Upload"
echo "========================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed"
    echo "Please install Git first"
    exit 1
fi

# Initialize git repository
echo "[1/5] Initializing Git repository..."
git init

# Add all files
echo "[2/5] Adding files to Git..."
git add .

# Commit
echo "[3/5] Committing files..."
git commit -m "Initial commit: Kibble pet food segmentation and super-resolution pipeline"

# Set branch to main
echo "[4/5] Setting branch to main..."
git branch -M main

echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo "1. Create a new repository on GitHub (if not already created)"
echo "2. Copy the repository URL"
echo "3. Run the following commands (replace URL with your repository URL):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "Or use SSH:"
echo "   git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "========================================"

