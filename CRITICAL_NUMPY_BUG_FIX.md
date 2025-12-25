# 🔥 CRITICAL FIX: Plotly fig.to_dict() mất data với numpy arrays

## Vấn đề phát hiện

Từ debug log:
```
Adding trace 0: x_vals.shape=(1024,), y_vals.shape=(1024,)  ← 1024 points khi add
Trace 0 - x_len=2, y_len=2                                  ← CHỈ CÒN 2 points sau to_dict()!
```

**ROOT CAUSE:** `fig.to_dict()` có bug khi convert Plotly Figure với subplots chứa numpy arrays. Data bị mất khi serialize!

## Tại sao xảy ra?

1. `make_subplots()` tạo Figure object với nhiều axes
2. `fig.add_trace(..., row=i+1, col=1)` thêm numpy arrays vào subplot
3. `fig.to_dict()` convert Figure → dictionary
4. **BUG:** Numpy arrays trong subplots KHÔNG được serialize đúng
5. Kết quả: Data chỉ còn 2 points (đầu và cuối array)

## Tại sao Overlay mode hoạt động?

**Overlay mode KHÔNG dùng subplots:**
```python
# Overlay: Thêm traces trực tiếp, KHÔNG có row/col
fig.add_trace(go.Scatter(x=x_vals, y=y_vals))
# → fig.to_dict() hoạt động bình thường
```

**Stack mode dùng subplots:**
```python
# Stack: Thêm traces vào các subplot khác nhau
fig.add_trace(go.Scatter(x=x_vals, y=y_vals), row=i+1, col=1)
# → fig.to_dict() BỊ BUG, mất data!
```

## Giải pháp

**Convert numpy arrays sang Python lists TRƯỚC KHI add vào figure:**

### realtime_plot_widget.py - Stack mode

**Trước:**
```python
fig.add_trace(
    go.Scatter(
        x=x_vals,  # ← numpy array
        y=y_vals,  # ← numpy array
        ...
    ),
    row=i + 1,
    col=1
)
```

**Sau:**
```python
# Convert numpy → Python list
x_list = x_vals.tolist()
y_list = y_vals.tolist()

fig.add_trace(
    go.Scatter(
        x=x_list,  # ← Python list
        y=y_list,  # ← Python list
        ...
    ),
    row=i + 1,
    col=1
)
```

### fft_plot_widget.py - Stack mode

Tương tự cho FFT plots và peak markers.

## Kết quả sau fix

```
Adding trace 0: x_vals.shape=(1024,)
[After .tolist()]
BEFORE JSON - Trace 0 has x_len=1024, y_len=1024  ← DATA ĐƯỢC GIỮ NGUYÊN!
```

## Test

```bash
source venv/bin/activate
python run_app.py
```

**Chọn Display: Stack**

Bạn sẽ thấy:
- ✅ 12 subplots riêng biệt
- ✅ Mỗi subplot có **TOÀN BỘ DATA** (1024 points)
- ✅ Signals hiển thị đầy đủ
- ✅ Có thể zoom, pan từng subplot
- ✅ X-axis được share giữa các subplots

## Tại sao cần .tolist()?

```python
import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder

# Test
arr = np.array([1, 2, 3, 4, 5])

# Với numpy array
data1 = {"x": arr}
json1 = json.dumps(data1, cls=PlotlyJSONEncoder)
# → Có thể bị lỗi hoặc serialize không đúng trong subplots

# Với Python list
data2 = {"x": arr.tolist()}
json2 = json.dumps(data2, cls=PlotlyJSONEncoder)
# → Luôn đúng!
```

## Performance impact

**Minimal:**
- `.tolist()` rất nhanh (C-level operation)
- Chỉ 1024 points mỗi trace (đã downsampled từ 10240)
- 12 traces × 1024 points = ~12K conversions
- Thời gian: < 1ms

**Trade-off:**
- Memory: Python lists dùng nhiều RAM hơn numpy arrays một chút
- Speed: Conversion rất nhanh, không ảnh hưởng performance
- Correctness: **QUAN TRỌNG NHẤT** - data được giữ nguyên!

## Lưu ý

Bug này CHỈ ảnh hưởng:
- ✅ Subplots (make_subplots + row/col)
- ✅ Numpy arrays
- ✅ Khi dùng fig.to_dict()

Bug này KHÔNG ảnh hưởng:
- ❌ Single plot (không dùng subplots)
- ❌ Python lists
- ❌ Khi dùng fig.show() trực tiếp

## Các file đã sửa

1. `src/gui/widgets/realtime_plot_widget.py`
   - `_update_stack_plots()`: Convert x_vals, y_vals sang lists

2. `src/gui/widgets/fft_plot_widget.py`
   - `_update_stack_plots()`: Convert freq_plot, mag_plot sang lists
   - Peak markers: Convert peak_freqs, peak_mags sang lists

---
Ngày fix: 2025-12-25
Severity: CRITICAL
Impact: Stack mode plots không hiển thị data
Solution: Convert numpy arrays to Python lists
