@echo off
cd /d "%~dp0"
"C:\Users\PC\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run stock_forecast_app/app.py --server.port 8501 --server.address 127.0.0.1
pause
