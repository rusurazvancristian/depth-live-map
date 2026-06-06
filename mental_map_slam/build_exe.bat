@echo off
cd /d D:\mental_map_slam
echo.
echo ============================================================
echo   Mental Map SLAM — Building Pitch Demo EXE
echo ============================================================
echo.

echo [1/3] Generating thumbnails...
python -c "
import cv2, sys
from pathlib import Path
out = Path('output')
td  = Path('pitch_demo/thumbs')
td.mkdir(parents=True, exist_ok=True)
for i in range(1,6):
    dst = td / f'video_{i}_thumb.jpg'
    if dst.exists(): continue
    src = out / f'video_{i}_demo.mp4'
    if not src.exists(): continue
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, f = cap.read()
    if ret: cv2.imwrite(str(dst), f, [cv2.IMWRITE_JPEG_QUALITY,88])
    cap.release()
print('Thumbnails OK')
"

echo.
echo [2/3] Running PyInstaller...
pyinstaller --noconfirm --onedir --windowed ^
    --name "MentalMapSLAM_Demo" ^
    --add-data "pitch_demo\ui.html;pitch_demo" ^
    --add-data "pitch_demo\thumbs;pitch_demo\thumbs" ^
    --add-data "output\demo_3d.html;output" ^
    --add-data "output\video_1_demo.mp4;output" ^
    --add-data "output\video_2_demo.mp4;output" ^
    --add-data "output\video_3_demo.mp4;output" ^
    --add-data "output\video_4_demo.mp4;output" ^
    --add-data "output\video_5_demo.mp4;output" ^
    --add-data "output\all_maps.png;output" ^
    --hidden-import "webview.platforms.edgechromium" ^
    --hidden-import "webview.platforms.winforms" ^
    "pitch_demo\launcher.py"

echo.
echo [3/3] Done!
echo.
echo   EXE location:
echo   dist\MentalMapSLAM_Demo\MentalMapSLAM_Demo.exe
echo.
echo   To distribute: zip the entire dist\MentalMapSLAM_Demo\ folder.
echo   Requires: Windows 10/11 with Microsoft Edge (WebView2) installed.
echo   (WebView2 is pre-installed on all Windows 11 machines.)
echo.
pause
