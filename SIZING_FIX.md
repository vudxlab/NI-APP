# ✅ Fix vấn đề sizing - Plot bị squeeze sang trái

## Vấn đề

Từ screenshot @2025-12-25_18-59.png:
- ✅ Plots ĐÃ HIỂN THỊ (thành công!)
- ❌ Plots chỉ chiếm ~40% width của container
- ❌ Phần bên phải bị trống
- ❌ Plot bị "squeezed" sang bên trái

## Nguyên nhân

1. **PlotlyView không expand**: Size policy mặc định không fill parent
2. **Plotly autosize không được bật**: Layout thiếu `autosize: true`
3. **Fixed width/height**: Plotly sử dụng size cố định thay vì responsive
4. **Không có resize event**: Plotly không biết khi container thay đổi size

## Các thay đổi đã thực hiện

### 1. PlotlyView - Set size policy to Expanding

**File:** `src/gui/widgets/plotly_view.py`

```python
# Set size policy to expand and fill parent
from PyQt5.QtWidgets import QSizePolicy
self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
```

### 2. Force Plotly autosize trong JavaScript

**File:** `src/gui/widgets/plotly_view.py`

```javascript
window.updatePlot = function(data, layout, config) {
    // Force layout to use full width and height
    if (!layout.width) layout.width = null;
    if (!layout.height) layout.height = null;
    if (!layout.autosize) layout.autosize = true;

    Plotly.react('plot', data, layout, config).then(function() {
        // Force a resize after initial render
        window.dispatchEvent(new Event('resize'));
    });
};
```

### 3. Thêm autosize vào base_layout

**Files:**
- `src/gui/widgets/realtime_plot_widget.py`
- `src/gui/widgets/fft_plot_widget.py`

```python
def _base_layout(self, y_title: str, x_title: str, show_legend: bool) -> Dict:
    return {
        "autosize": True,  # ← THÊM DÒNG NÀY
        "margin": {"l": 60, "r": 20, "t": 20, "b": 45},
        # ... rest of layout
        "xaxis": {
            "automargin": True,  # ← THÊM DÒNG NÀY
            # ...
        },
        "yaxis": {
            "automargin": True,  # ← THÊM DÒNG NÀY
            # ...
        }
    }
```

### 4. Update fig.update_layout cho stack mode

```python
fig.update_layout(
    autosize=True,  # ← THÊM DÒNG NÀY
    margin={"l": 60, "r": 20, "t": 10, "b": 45},
    # ...
)
```

## Kết quả mong đợi

Sau khi chạy app lại:

```bash
source venv/bin/activate
python run_app.py
```

Bạn sẽ thấy:
- ✅ Plots FILL TOÀN BỘ width của container
- ✅ Data hiển thị từ đầu đến cuối timeline
- ✅ Không còn khoảng trống bên phải
- ✅ Plot resize khi bạn thay đổi kích thước window

## Test

1. **Chạy app**
2. **Resize window** - Plot sẽ tự động resize theo
3. **Switch giữa tabs** - Plots luôn fill toàn bộ space
4. **Switch giữa Overlay/Stack mode** - Sizing đúng cho cả 2 modes

## Debug

Nếu vẫn có vấn đề, kiểm tra console:
```
Plot updated successfully
```

Nếu thấy message này → Plotly đã render thành công.

## Lưu ý

- Plotly responsive mode chỉ hoạt động khi `autosize: true`
- Widget cần `Expanding` size policy để fill parent
- `window.dispatchEvent(new Event('resize'))` force Plotly recalculate size

---
Ngày fix: 2025-12-25
