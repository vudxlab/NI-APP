# Fix vấn đề width - Plots chỉ chiếm 50% container

## Vấn đề

Tất cả plots (Overlay và Stack) chỉ chiếm ~50% width của container, phần còn lại bị trống.

## Nguyên nhân

1. **PlotlyView không resize:** Khi container resize, PlotlyView không trigger Plotly relayout
2. **Layout width bị fix:** Plotly có thể cache width ban đầu
3. **Không có resize handler:** Không có code để handle resize events

## Giải pháp

### 1. Thêm resizeEvent handler

**File:** `src/gui/widgets/plotly_view.py`

```python
def resizeEvent(self, event):
    """Handle resize events and trigger Plotly relayout."""
    super().resizeEvent(event)
    # Trigger Plotly to relayout after resize
    if self._loaded and self.page():
        resize_script = """
        if (typeof Plotly !== 'undefined') {
            Plotly.Plots.resize('plot');
        }
        """
        self.page().runJavaScript(resize_script)
```

### 2. Force width = undefined trong updatePlot

**Trước:**
```javascript
if (!layout.width) layout.width = null;
```

**Sau:**
```javascript
layout.width = undefined;  // Force Plotly to calculate width
```

### 3. Thêm Plotly.Plots.resize() sau update

```javascript
Plotly.react('plot', data, layout, config).then(function() {
    setTimeout(function() {
        Plotly.Plots.resize('plot');  // Force recalculate
    }, 100);
});
```

### 4. Cải thiện CSS

**Thêm:**
```css
* {
    box-sizing: border-box;  /* Ensure proper sizing */
}
```

**Thêm meta viewport:**
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

## Cách hoạt động

### Flow bình thường:
```
Container resize → PlotlyView resize → [KHÔNG CÓ GÌ XẢY RA] → Plot vẫn 50% width
```

### Flow sau fix:
```
Container resize → PlotlyView.resizeEvent() → Plotly.Plots.resize('plot') → Plot fill 100% width
```

## Test

```bash
source venv/bin/activate
python run_app.py
```

**Test cases:**
1. **Resize window:** Kéo window lớn hơn/nhỏ hơn → Plot tự động resize
2. **Switch tabs:** Chuyển giữa Time Domain / Frequency Domain → Plot fill width
3. **Switch mode:** Overlay ↔ Stack → Plot fill width
4. **Splitter:** Kéo splitter giữa config panel và plot → Plot resize theo

## Kết quả mong đợi

**Trước:**
```
┌──────────────────────────────────────────┐
│ ████████████████████                     │  ← Plot chỉ 50%
│                                          │
└──────────────────────────────────────────┘
```

**Sau:**
```
┌──────────────────────────────────────────┐
│ ████████████████████████████████████████ │  ← Plot 100%
│                                          │
└──────────────────────────────────────────┘
```

## Debug

Nếu vẫn không hoạt động, check console:
```javascript
// Trong browser console (F12)
Plotly.Plots.resize('plot');  // Manual trigger
```

Hoặc check PlotlyView size:
```python
print(f"PlotlyView size: {self.plot_view.size()}")
print(f"PlotlyView width: {self.plot_view.width()}")
```

## Lưu ý

- `Plotly.Plots.resize()` là API chính thức của Plotly để trigger relayout
- `resizeEvent()` được gọi tự động khi Qt widget resize
- `setTimeout(100ms)` đảm bảo DOM đã stable trước khi resize

---
Ngày fix: 2025-12-25
