@echo off
echo Setting up AskMyNotes project...

echo.
echo Setting up Python backend...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Setting up React frontend...
cd ..\frontend
npm install

echo.
echo Setup complete! You can now:
echo 1. Start the backend: cd backend ^& venv\Scripts\activate ^& uvicorn main:app --reload
echo 2. Start the frontend: cd frontend ^& npm run dev 