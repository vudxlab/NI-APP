# Tóm tắt sửa lỗi Plotly View

## Vấn đề
Biểu đồ không xuất hiện trong các tab "Time Domain" và "Frequency Domain" sau khi chuyển đổi từ plotting thông thường sang Plotly với QtWebEngine.

## Nguyên nhân
Có 3 vấn đề chính:

### 1. Timing Issue (Vấn đề về thời gian)
- PlotlyView khởi tạo HTML bất đồng bộ (asynchronously)
- Các widget plot gọi `update_plot()` ngay lập tức sau khi tạo PlotlyView
- Điều này xảy ra TRƯỚC KHI HTML hoàn tất việc load
- Mặc dù có cơ chế `_pending_payload`, nhưng có thể gặp vấn đề race condition

### 2. Thiếu error handling
- PlotlyView không có error handling đầy đủ
- Không có log khi HTML load thất bại
- Không kiểm tra `page()` có thể là None

### 3. Layout structure
- Tạo QVBoxLayout trung gian không cần thiết (`plot_container`)
- Có thể gây vấn đề về sizing/visibility của widget

## Giải pháp

### File: `src/gui/widgets/plotly_view.py`
**Thay đổi:**
1. Thêm minimum size cho widget (400x300) để đảm bảo hiển thị
2. Thêm error handling trong `_on_load_finished()`
3. Thêm null check cho `page()` trong `_run_payload()`
4. Thêm callback `_on_script_result()` để debug JavaScript

```python
def __init__(self, parent=None):
    super().__init__(parent)
    self._loaded = False
    self._pending_payload: Optional[str] = None
    self.loadFinished.connect(self._on_load_finished)

    # Set minimum size to ensure widget is visible
    self.setMinimumSize(400, 300)

    self._init_html()
```

### File: `src/gui/widgets/realtime_plot_widget.py`
**Thay đổi:**
1. Loại bỏ `plot_container` layout trung gian
2. Thêm PlotlyView trực tiếp vào main layout
3. Trì hoãn khởi tạo empty plot bằng `QTimer.singleShot(100ms)`
4. Thêm method `_init_empty_plot()` để khởi tạo plot sau khi widget visible

```python
if PLOTLY_AVAILABLE:
    self.plot_view = PlotlyView()
    layout.addWidget(self.plot_view)
    # Initial empty plot will be set after widget is shown
    QTimer.singleShot(100, self._init_empty_plot)
```

### File: `src/gui/widgets/fft_plot_widget.py`
**Thay đổi:** Tương tự như `realtime_plot_widget.py`

### File: `src/gui/widgets/long_fft_widget.py`
**Thay đổi:**
1. Import QTimer
2. Áp dụng fix tương tự như các widget khác

## Kiểm tra

Chạy script test:
```bash
source venv/bin/activate
python test_plotly_view.py
```

Script này sẽ:
1. Tạo window với 2 tabs (Time Domain và Frequency Domain)
2. Configure cả 2 widgets
3. Gửi test data sau 500ms
4. Kiểm tra xem plots có hiển thị không

## Lưu ý
- Thời gian delay 100ms là đủ cho hầu hết các trường hợp
- Nếu vẫn gặp vấn đề, có thể tăng delay lên 200ms hoặc 500ms
- Kiểm tra console output để xem các error messages từ PlotlyView
- Đảm bảo PyQtWebEngine đã được cài đặt: `pip install PyQtWebEngine>=5.15.0`

## Files đã sửa đổi
1. `src/gui/widgets/plotly_view.py`
2. `src/gui/widgets/realtime_plot_widget.py`
3. `src/gui/widgets/fft_plot_widget.py`
4. `src/gui/widgets/long_fft_widget.py`

## Ngày sửa
2025-12-25
