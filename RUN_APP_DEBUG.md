# Hướng dẫn chạy app với debug mode

## Chạy test script

Để kiểm tra nhanh xem Plotly plots có hoạt động không:

```bash
source venv/bin/activate
python test_plotly_view.py
```

**Kết quả mong đợi:**
- Window mở với 2 tabs (Time Domain và Frequency Domain)
- Sau ~500ms: cả 2 tabs hiển thị lưới trống với trục tọa độ (màu nền #f5f5f5)
- Sau ~1s: tab Time Domain hiển thị 4 đường sine wave
- Console hiển thị nhiều debug messages từ PlotlyView

**Debug messages quan trọng:**
```
PlotlyView.__init__: Creating PlotlyView...
PlotlyView._init_html: Loading Plotly.js...
PlotlyView._init_html: Plotly.js loaded, size=XXXXXX bytes
PlotlyView._on_load_finished: ok=True, has_pending=False
PlotlyView._on_load_finished: HTML loaded successfully!
```

## Chạy app chính

```bash
source venv/bin/activate
python run_app.py
```

hoặc

```bash
source venv/bin/activate
python src/main.py
```

## Kiểm tra các tab

1. **Tab "Time Domain":**
   - Nên thấy lưới trống với labels "Time (s)" và "Acceleration (g)"
   - Nếu có data, sẽ thấy đường plot

2. **Tab "Frequency Domain":**
   - Nên thấy lưới trống với labels "Frequency (Hz)" và "Magnitude (dB)"
   - Click "Analyze Saved Data" để xem FFT

## Nếu vẫn không thấy plots

### Kiểm tra console output

Tìm các dòng này trong console:

```
PlotlyView.__init__: Creating PlotlyView, size=...
PlotlyView._init_html: Loading Plotly.js...
PlotlyView._init_html: Plotly.js loaded, size=... bytes
```

**Nếu KHÔNG thấy:** PlotlyView không được khởi tạo
- Kiểm tra PLOTLY_AVAILABLE = True trong widget files

**Nếu thấy nhưng KHÔNG có "HTML loaded successfully!":**
- HTML load thất bại
- Có thể thiếu QtWebEngine
- Chạy: `pip install PyQtWebEngine`

**Nếu thấy "HTML loaded successfully!" nhưng không có plot:**
- Tăng delay time từ 500ms lên 1000ms trong các widget files
- Sửa dòng `QTimer.singleShot(500, ...)` thành `QTimer.singleShot(1000, ...)`

### Kiểm tra PyQtWebEngine

```bash
source venv/bin/activate
python -c "from PyQt5.QtWebEngineWidgets import QWebEngineView; print('OK')"
```

Nếu lỗi:
```bash
pip install PyQtWebEngine>=5.15.0
```

### Kiểm tra Plotly.js size

Plotly.js rất lớn (~3-4 MB). Nếu máy chậm, cần tăng delay:

```bash
source venv/bin/activate
python -c "from plotly.offline import get_plotlyjs; print(f'Plotly.js size: {len(get_plotlyjs())/1024/1024:.2f} MB')"
```

## Tắt debug messages

Sau khi app hoạt động, có thể xóa các dòng `print()` trong:
- `src/gui/widgets/plotly_view.py`
- `src/gui/widgets/realtime_plot_widget.py`
- `src/gui/widgets/fft_plot_widget.py`

Hoặc comment chúng lại để dễ debug sau này.

## Lưu ý

- Delay 500ms là đủ cho hầu hết các máy
- Nếu máy chậm, có thể cần 1000ms hoặc 2000ms
- PlotlyView cần QtWebEngine để hoạt động
- Không thể dùng PyQtGraph thay thế cho Plotly interactive plots
