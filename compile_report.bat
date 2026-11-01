@echo off
setlocal
echo ==============================================================================
echo DANG BIEN DICH BAO CAO LATEX OFFLINE QUA MIKTEX (KHONG CAN OVERLEAF)
echo ==============================================================================

set "PATH=C:\Users\thefu\AppData\Local\Programs\MiKTeX\miktex\bin\x64;%PATH%"

cd /d "%~dp0docs\overleaf_report"

echo [1/2] Dang chay pdflatex (luot 1)...
pdflatex -interaction=nonstopmode main.tex > nul

echo [2/2] Dang chay pdflatex (luot 2 - cap nhat muc luc va tham chieu)...
pdflatex -interaction=nonstopmode main.tex > nul

if exist "main.pdf" (
    echo.
    echo ==============================================================================
    echo BIEN DICH THANH CONG! File PDF da san sang:
    echo %~dp0docs\overleaf_report\main.pdf
    echo ==============================================================================
    
    copy /y "main.pdf" "%USERPROFILE%\Downloads\Bao_Cao_Tot_Nghiep.pdf" > nul
    echo Da tu dong copy sang: %USERPROFILE%\Downloads\Bao_Cao_Tot_Nghiep.pdf
    
    start "" "main.pdf"
) else (
    echo [!] Co loi xay ra trong qua trinh bien dich. Vui long kiem tra main.log!
)

endlocal
pause
