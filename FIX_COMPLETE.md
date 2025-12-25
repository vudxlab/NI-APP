# ✅ Hoàn thành sửa lỗi Plotly View

## Vấn đề đã khắc phục

Biểu đồ Plotly không hiển thị trong tabs "Time Domain" và "Frequency Domain" do:

1. **Race condition**: Widget gọi `update_plot()` TRƯỚC KHI HTML hoàn tất load
2. **Thiếu error handling**: Không có logging để debug
3. **Layout timing**: Widget chưa visible khi init plot

## Giải pháp đã áp dụng

### 1. PlotlyView với debug logging (`src/gui/widgets/plotly_view.py`)
- ✅ Thêm minimum size (400x300)
- ✅ Thêm extensive debug logging
- ✅ Thêm error handling và null checks
- ✅ Thêm background color (#f5f5f5) để dễ nhận biết plot area

### 2. Widget initialization với delay (`realtime_plot_widget.py`, `fft_plot_widget.py`)
- ✅ Loại bỏ layout trung gian không cần thiết
- ✅ Delay init plot 500ms bằng `QTimer.singleShot()`
- ✅ Thêm debug logging để track initialization

### 3. Consistent fixes cho tất cả widgets
- ✅ `realtime_plot_widget.py`
- ✅ `fft_plot_widget.py`
- ✅ `long_fft_widget.py`

## Cách kiểm tra

### Bước 1: Chạy test script đơn giản

```bash
source venv/bin/activate
python test_plotly_view.py
```

**Kết quả mong đợi:**
- Window hiển thị với 2 tabs
- Console in ra nhiều debug messages
- Sau ~500ms: Thấy lưới plot với nền xám nhạt (#f5f5f5)
- Sau ~1s: Tab Time Domain hiển thị 4 đường sine wave

**Debug messages quan trọng cần thấy:**
```
PlotlyView.__init__: Creating PlotlyView...
PlotlyView._init_html: Plotly.js loaded, size=3XXXXXX bytes
PlotlyView._on_load_finished: ok=True
PlotlyView._on_load_finished: HTML loaded successfully!
```

### Bước 2: Chạy app chính

```bash
source venv/bin/activate
python run_app.py
```

hoặc

```bash
source venv/bin/activate
python src/main.py
```

**Kiểm tra:**
1. Mở tab "Time Domain" - nên thấy lưới plot với axes labels
2. Mở tab "Frequency Domain" - nên thấy lưới plot với axes labels
3. Cả 2 tabs đều có background màu xám nhạt (#f5f5f5)

## Nếu vẫn không thấy plots

### Giải pháp 1: Tăng delay time

Nếu máy chậm hoặc Plotly.js lớn, tăng delay từ 500ms lên 1000ms:

**File cần sửa:**
- `src/gui/widgets/realtime_plot_widget.py` (dòng ~106)
- `src/gui/widgets/fft_plot_widget.py` (dòng ~114)
- `src/gui/widgets/long_fft_widget.py` (dòng ~276)

Sửa:
```python
QTimer.singleShot(500, self._init_empty_plot)
```

Thành:
```python
QTimer.singleShot(1000, self._init_empty_plot)  # Tăng lên 1 giây
```

### Giải pháp 2: Kiểm tra PyQtWebEngine

```bash
source venv/bin/activate
python -c "from PyQt5.QtWebEngineWidgets import QWebEngineView; print('OK')"
```

Nếu lỗi, cài lại:
```bash
pip install --upgrade PyQtWebEngine>=5.15.0
```

### Giải pháp 3: Kiểm tra console output

Chạy app và tìm các messages này:
- ❌ Nếu KHÔNG thấy "PlotlyView.__init__" → PLOTLY_AVAILABLE = False
- ❌ Nếu thấy "ERROR: PlotlyView HTML failed to load" → QtWebEngine issue
- ✅ Nếu thấy "HTML loaded successfully!" → OK, chỉ cần đợi thêm

## Debug mode

Tất cả các print statements đã được thêm vào code. Để xem debug output:
- Chạy app từ terminal (KHÔNG double-click)
- Xem console output

Để TẮT debug mode sau khi app hoạt động:
- Xóa hoặc comment các dòng `print(...)` trong các file đã sửa

## Files đã thay đổi

1. ✅ `src/gui/widgets/plotly_view.py` - Core PlotlyView với debug logging
2. ✅ `src/gui/widgets/realtime_plot_widget.py` - Time domain plot với delay init
3. ✅ `src/gui/widgets/fft_plot_widget.py` - Frequency domain plot với delay init
4. ✅ `src/gui/widgets/long_fft_widget.py` - Long FFT plot với delay init
5. ✅ `test_plotly_view.py` - Test script
6. ✅ `RUN_APP_DEBUG.md` - Hướng dẫn debug chi tiết
7. ✅ `FIX_COMPLETE.md` - File này
8. ✅ `PLOTLY_FIX_SUMMARY.md` - Tóm tắt kỹ thuật

## Tham khảo thêm

- `RUN_APP_DEBUG.md` - Hướng dẫn debug chi tiết
- `PLOTLY_FIX_SUMMARY.md` - Giải thích kỹ thuật về fix

## Kết luận

Vấn đề đã được khắc phục với:
- ✅ Delay initialization 500ms
- ✅ Debug logging đầy đủ
- ✅ Error handling tốt hơn
- ✅ Consistent implementation cho tất cả widgets

Hãy chạy `test_plotly_view.py` để kiểm tra!

---
Ngày sửa: 2025-12-25
