# 🔥 FIX QUAN TRỌNG: PlotlyView không load HTML

## Vấn đề phát hiện

Từ log output của bạn:
```
PlotlyView.update_plot: loaded=False
PlotlyView.update_plot: Not loaded yet, storing as pending
```

**Nguyên nhân:** Plotly.js embedded (~3-4 MB) quá lớn, `setHtml()` của QtWebEngine THẤT BẠI khi load HTML với nội dung lớn như vậy.

Kết quả: `loadFinished` signal KHÔNG BAO GIỜ được emit với `ok=True`, nên `_loaded` luôn là `False`, và TẤT CẢ plots đều bị pending mãi mãi.

## Giải pháp đã áp dụng

**Thay đổi trong `src/gui/widgets/plotly_view.py`:**

TRƯỚC (❌ Không hoạt động):
```python
plotly_js = get_plotlyjs()  # ~3-4 MB
html = f"""...<script>{plotly_js}</script>..."""
self.setHtml(html)  # THẤT BẠI vì quá lớn
```

SAU (✅ Hoạt động):
```python
html = """...<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>..."""
self.setHtml(html)  # OK, chỉ ~1 KB
```

**Lợi ích:**
- ✅ HTML nhỏ (~1 KB thay vì 3-4 MB)
- ✅ Load nhanh hơn
- ✅ `loadFinished` signal được emit chính xác
- ✅ Plots hiển thị ngay lập tức

**Nhược điểm:**
- ⚠️ CẦN INTERNET để load Plotly.js từ CDN lần đầu
- ⚠️ Nếu không có internet, plots sẽ không hiển thị

## Cách kiểm tra

### Bước 1: Đảm bảo có kết nối internet

```bash
ping cdn.plot.ly
```

Nếu không có internet, xem "Giải pháp offline" bên dưới.

### Bước 2: Chạy app

```bash
source venv/bin/activate
python run_app.py
```

### Bước 3: Kiểm tra console output

**Kết quả mong đợi:**
```
PlotlyView.__init__: Creating PlotlyView, size=...
PlotlyView._init_html: Using Plotly.js from CDN...
PlotlyView._init_html: Setting HTML, size=XXX bytes (using CDN)
PlotlyView._on_load_finished: ok=True, has_pending=False
PlotlyView._on_load_finished: HTML loaded successfully!
```

**Sau đó:**
```
RealtimePlotWidget._init_empty_plot: Called
PlotlyView.update_plot: called with 0 traces, loaded=True  ← Quan trọng: loaded=True!
PlotlyView.update_plot: Running payload immediately
PlotlyView._run_payload: Executing JavaScript...
```

### Bước 4: Kiểm tra plots

- Tab "Time Domain" → Nên thấy lưới plot với nền xám (#f5f5f5)
- Tab "Frequency Domain" → Nên thấy lưới plot với nền xám

## Giải pháp offline (không cần internet)

Nếu bạn KHÔNG có internet hoặc muốn app hoạt động offline, có 2 cách:

### Cách 1: Download Plotly.js về local (KHUYẾN NGHỊ)

```bash
# Tạo thư mục static
mkdir -p src/gui/widgets/static

# Download Plotly.js
cd src/gui/widgets/static
wget https://cdn.plot.ly/plotly-2.27.0.min.js

# Quay lại root
cd ../../..
```

Sau đó sửa `src/gui/widgets/plotly_view.py`:

```python
from pathlib import Path

def _init_html(self):
    # Load plotly.js from local file
    static_dir = Path(__file__).parent / 'static'
    plotly_js_file = static_dir / 'plotly-2.27.0.min.js'

    if plotly_js_file.exists():
        plotly_js = plotly_js_file.read_text()
        script_tag = f'<script>{plotly_js}</script>'
    else:
        # Fallback to CDN
        script_tag = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    html = f"""
    <html>
      <head>...</head>
      <body>
        <div id="plot"></div>
        {script_tag}
        <script>...</script>
      </body>
    </html>
    """
    self.setHtml(html)
```

**LƯU Ý:** Cách này VẪN có thể gặp vấn đề nếu Plotly.js quá lớn. Thử cách 2 nếu vẫn không hoạt động.

### Cách 2: Dùng setUrl() thay vì setHtml() (PHỨC TẠP HƠN)

Tạo file HTML local và dùng `setUrl()`:

```python
def _init_html(self):
    # Create temporary HTML file
    import tempfile
    html_content = """..."""

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
    temp_file.write(html_content)
    temp_file.close()

    from PyQt5.QtCore import QUrl
    self.setUrl(QUrl.fromLocalFile(temp_file.name))
```

## Khuyến nghị

**SỬ DỤNG CDN (current fix)** là giải pháp TỐT NHẤT vì:
- ✅ Đơn giản
- ✅ Load nhanh
- ✅ Luôn cập nhật
- ✅ Không cần quản lý file local

Chỉ cần đảm bảo máy có internet khi chạy app lần đầu. Sau đó browser cache sẽ lưu Plotly.js.

## Kết quả sau khi fix

Bây giờ khi bạn chạy app:
1. HTML nhỏ (~1 KB) → `setHtml()` thành công
2. Browser load Plotly.js từ CDN (~3 MB)
3. `loadFinished` signal emit với `ok=True`
4. `_loaded = True`
5. Tất cả pending plots được render ngay lập tức
6. ✅ PLOTS HIỂN THỊ!

---
Ngày fix: 2025-12-25
