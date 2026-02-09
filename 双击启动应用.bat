@echo off
:: 设置代码页为 UTF-8 以解决中文乱码问题
chcp 65001 >nul
SETLOCAL
SET WORKDIR=%~dp0
cd /d %WORKDIR%

echo ==========================================
echo    正在启动 Local-Multi-RAG 系统...
echo ==========================================

:: 检查常见的虚拟环境名称
if exist ".venv-1\Scripts\python.exe" (
    set PY_PATH=.venv-1\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PY_PATH=venv\Scripts\python.exe
) else (
    echo [错误] 未找到虚拟环境！请先运行 pip install -r requirements.txt。
    pause
    exit /b
)

echo 使用环境: %PY_PATH%
%PY_PATH% -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo [错误] 应用运行出错。
    pause
)

ENDLOCAL
