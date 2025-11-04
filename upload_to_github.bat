@echo off
REM Quick script to upload Colgate project to GitHub
REM Usage: Run this script after creating a GitHub repository

echo ========================================
echo Colgate Kibble Segmentation - GitHub Upload
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

REM Initialize git repository
echo [1/5] Initializing Git repository...
git init
if %errorlevel% neq 0 (
    echo ERROR: Failed to initialize git repository
    pause
    exit /b 1
)

REM Add all files
echo [2/5] Adding files to Git...
git add .
if %errorlevel% neq 0 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)

REM Commit
echo [3/5] Committing files...
git commit -m "Initial commit: Kibble pet food segmentation and super-resolution pipeline"
if %errorlevel% neq 0 (
    echo WARNING: Commit failed. This might be normal if no changes to commit.
)

REM Set branch to main
echo [4/5] Setting branch to main...
git branch -M main

echo.
echo ========================================
echo Next steps:
echo ========================================
echo 1. Create a new repository on GitHub (if not already created)
echo 2. Copy the repository URL
echo 3. Run the following command (replace URL with your repository URL):
echo.
echo    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
echo    git push -u origin main
echo.
echo Or use SSH:
echo    git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
echo    git push -u origin main
echo.
echo ========================================
pause

