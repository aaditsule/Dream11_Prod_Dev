# Dream Team Predictor

A full-stack web application that predicts fantasy cricket points for IPL players and recommends an optimal Dream11 team based on a variety of constraints.

---

## Overview

This project provides an end-to-end solution for fantasy cricket enthusiasts. It ingests raw, ball-by-ball match data, processes it to generate historical features, predicts player performance using a machine learning model, and selects the best possible 11-player team using a constrained optimization solver. The results are displayed in a clean, interactive web interface.

---

## Tech Stack

- **Backend**: Python, FastAPI, Pandas, Scikit-learn, XGBoost, SHAP, PuLP  
- **Frontend**: React.js, Axios, JavaScript (ES6+), CSS3  
- **Database**: File-based (CSV for historical data)  

---

## Project Structure

```
.
├── backend/              # All Python backend logic
│   ├── data/             # Input data files
│   ├── model_artifacts/  # Saved ML model
│   ├── src/              # Source code for all modules
│   └── main.py           # FastAPI application
├── frontend/             # React frontend application

```

---

## Setup and Installation

### Prerequisites
- Python 3.10+  
- Node.js v16+ and npm  

---

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:

   **Bash / MacOS / Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the required Node.js packages:
   ```bash
   npm install
   ```

---

### Running the Application

The application requires both the backend and frontend servers to be running simultaneously.

**Start the Backend Server**
```bash
uvicorn main:app --reload
```
The API will be available at:  
[http://localhost:8000](http://localhost:8000)

**Start the Frontend Server**
```bash
npm start
```
The web application will open in your browser at:  
[http://localhost:3000](http://localhost:3000)

---

## How to Use

1. Open the web application in your browser.  
2. Click the **"Choose File"** button and select a valid match JSON file.  
3. Click **"Generate Team"**.  
4. The application will display the recommended 11-player team, along with:  
   - Total credits  
   - Predicted points  
   - Role/team composition  
5. Click **"Show Rationale"** on any player card to see a breakdown of the features that contributed to their predicted score.
