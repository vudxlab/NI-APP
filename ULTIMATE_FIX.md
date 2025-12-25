# 🚨 FIX CUỐI CÙNG: loadFinished signal không được gọi

## Vấn đề phát hiện từ log

Bạn vẫn thấy:
```
PlotlyView.update_plot: loaded=False
PlotlyView.update_plot: Not loaded yet, storing as pending
```

Nhưng KHÔNG THẤY:
```
PlotlyView._on_load_finished: ...
```

**Kết luận:** `loadFinished` signal KHÔNG BAO GIỜ được emit! QtWebEngine có vấn đề nghiêm trọng.

## Nguyên nhân có thể

1. **QtWebEngine chưa được khởi tạo đúng cách**
2. **Signal connection bị lỗi**
3. **HTML load thất bại im lặng (silent failure)**
4. **Thread/event loop issue**

## Giải pháp mới: Multi-stage loading

Thay vì load tất cả cùng lúc, giờ sẽ:

### Bước 1: Load HTML đơn giản (~1 KB)
```html
<div>Loading Plotly...</div>
```

### Bước 2: Đợi loadFinished
Nếu signal KHÔNG được gọi → BIẾT NGAY QtWebEngine bị lỗi

### Bước 3: Load Plotly.js động bằng JavaScript
```javascript
var script = document.createElement('script');
script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
script.onload = function() { /* ready */ };
```

### Bước 4: Polling để check Plotly ready
```javascript
window.plotReady === true
```

### Bước 5: Set _loaded = True
Chỉ khi Plotly.js THỰC SỰ sẵn sàng

## Debug messages quan trọng

Khi chạy app, bạn SẼ THẤY:

### Nếu QtWebEngine hoạt động:
```
PlotlyView.__init__: Creating PlotlyView...
PlotlyView._init_html: Creating simple HTML...
PlotlyView._on_load_started: *** LOAD STARTED ***  ← QUAN TRỌNG!
PlotlyView._on_load_finished: *** LOAD FINISHED *** ok=True  ← QUAN TRỌNG!
PlotlyView._on_load_finished: HTML loaded successfully! Now loading Plotly.js from CDN...
PlotlyView._on_plotly_load_script_executed: Script executed...
PlotlyView: Waiting 2 seconds for Plotly.js to load from CDN...
PlotlyView._check_plotly_ready: Checking if Plotly.js is ready...
PlotlyView._on_plotly_ready_checked: plotReady=True
PlotlyView: *** PLOTLY READY! *** Now processing pending payload...
PlotlyView: Running pending payload
PlotlyView._run_payload: Executing JavaScript...
```

### Nếu QtWebEngine KHÔNG hoạt động:
```
PlotlyView.__init__: Creating PlotlyView...
PlotlyView._init_html: Creating simple HTML...
PlotlyView._init_html: If you don't see '_on_load_started' and '_on_load_finished' messages, QtWebEngine is broken
[KHÔNG CÓ GÌ THÊM] ← VẤN ĐỀ Ở ĐÂY!
```

## Cách test

```bash
source venv/bin/activate
python run_app.py 2>&1 | grep -E "(PlotlyView|LOAD|PLOTLY READY)"
```

## Nếu vẫn KHÔNG thấy loadStarted/loadFinished

**Nghĩa là QtWebEngine CÓ VẤN ĐỀ CĂN BẢN!**

### Giải pháp 1: Cài lại PyQtWebEngine

```bash
source venv/bin/activate
pip uninstall PyQtWebEngine PyQt5
pip install PyQt5==5.15.10 PyQtWebEngine==5.15.6
```

### Giải pháp 2: Kiểm tra Qt platform plugin

```bash
export QT_DEBUG_PLUGINS=1
python run_app.py
```

Tìm errors liên quan đến QtWebEngine.

### Giải pháp 3: Kiểm tra dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libqt5webengine5 libqt5webenginecore5 libqt5webenginewidgets5

# Hoặc
ldd venv/lib/python*/site-packages/PyQt5/Qt5/lib/libQt5WebEngine*.so
```

### Giải pháp 4: Test QtWebEngine cơ bản

```bash
python -c "
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import sys

app = QApplication(sys.argv)
view = QWebEngineView()

def on_load_finished(ok):
    print(f'Load finished: {ok}')
    app.quit()

view.loadFinished.connect(on_load_finished)
view.setHtml('<html><body><h1>Test</h1></body></html>')
view.show()

sys.exit(app.exec_())
"
```

Nếu KHÔNG in ra "Load finished: True" → QtWebEngine bị lỗi hoàn toàn.

## Timeline mong đợi

- **T+0ms:** PlotlyView.__init__
- **T+10ms:** _on_load_started
- **T+50ms:** _on_load_finished (ok=True)
- **T+50ms:** Loading Plotly.js from CDN...
- **T+2050ms:** Checking if Plotly.js ready
- **T+2050ms:** plotReady=True
- **T+2050ms:** *** PLOTLY READY! ***
- **T+2050ms:** Running pending payload
- **→ PLOTS HIỂN THỊ!**

## Nếu thành công

Bạn sẽ thấy:
1. Text "Loading Plotly..." trong plot area
2. Sau 2 giây: Text "Ready for plots"
3. Sau đó: Plots thực sự hiển thị

## Quan trọng

**CHẠY APP VÀ PASTE TOÀN BỘ CONSOLE OUTPUT!**

Tôi cần thấy:
- ✅ Có "_on_load_started" không?
- ✅ Có "_on_load_finished" không?
- ✅ "ok=True" hay "ok=False"?

Điều này sẽ cho biết CHÍNH XÁC vấn đề ở đâu.

---
Ngày fix: 2025-12-25
