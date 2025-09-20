@echo off

REM 检查Python是否安装
where python >nul 2>nul
if %errorlevel% neq 0 (
echo 错误: 未找到Python。请先安装Python并将其添加到系统PATH。
pause
exit /b 1
)

REM 运行零件检测程序
python part_detection.py

REM 如果程序出错，显示错误信息并暂停
echo.
echo 程序已结束。
pause