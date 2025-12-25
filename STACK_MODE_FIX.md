# ✅ Fix Stack Mode - Subplots không hiển thị

## Vấn đề

Khi chọn Display: Stack, các subplot (biểu đồ con) KHÔNG hiển thị.

## Nguyên nhân

1. **Thiếu height**: Plotly subplots CẦN explicit height, nếu không sẽ collapse
2. **Vertical spacing quá nhỏ**: 0.02 → subplots bị chồng lên nhau
3. **Không có subplot titles**: Khó phân biệt các channels
4. **automargin chưa được set**: Axes labels bị cắt

## Giải pháp

### 1. Set explicit height dựa trên số channels

**Trước:**
```python
fig.update_layout(
    autosize=True,  # ← KHÔNG ĐỦ cho subplots!
    margin={"l": 60, "r": 20, "t": 10, "b": 45},
    ...
)
```

**Sau:**
```python
# Calculate height: 150px per subplot, minimum 400px
plot_height = max(400, self.n_channels * 150)

fig.update_layout(
    height=plot_height,  # ← QUAN TRỌNG!
    margin={"l": 60, "r": 20, "t": 40, "b": 45},  # Tăng top margin cho titles
    ...
)
```

### 2. Tăng vertical spacing

**Trước:**
```python
fig = make_subplots(
    rows=self.n_channels,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02  # ← QUÁ NHỎ!
)
```

**Sau:**
```python
fig = make_subplots(
    rows=self.n_channels,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,  # ← TỐT HƠN
    subplot_titles=[self.channel_names[i] for i in range(self.n_channels)]
)
```

### 3. Thêm automargin cho axes

```python
fig.update_yaxes(
    title_text=f"{self.channel_units}",
    automargin=True,  # ← QUAN TRỌNG
    row=i + 1,
    col=1
)

fig.update_xaxes(
    title_text="Time (s)",
    automargin=True,  # ← QUAN TRỌNG
    row=self.n_channels,
    col=1
)
```

### 4. Simplify y-axis titles

**Trước:**
```python
title_text=f"{self.channel_names[i]} ({self.channel_units})"
# → Quá dài, bị cắt
```

**Sau:**
```python
title_text=f"{self.channel_units}"
# Channel name đã có trong subplot_titles rồi
```

## Kết quả

### Với 4 channels:
- **Total height:** max(400, 4 * 150) = **600px**
- **Per subplot:** ~150px mỗi subplot
- **Spacing:** 5% vertical spacing giữa subplots
- **Titles:** Channel names ở top của mỗi subplot

### Với 12 channels:
- **Total height:** max(400, 12 * 150) = **1800px**
- **Per subplot:** ~150px mỗi subplot
- **Scrollable:** Container sẽ có scrollbar nếu cần

## Test

```bash
source venv/bin/activate
python run_app.py
```

**Các bước:**
1. Chọn Display: **Stack**
2. Bạn sẽ thấy:
   - ✅ Mỗi channel có subplot riêng
   - ✅ Channel names ở top mỗi subplot
   - ✅ Các subplots có khoảng cách hợp lý
   - ✅ Y-axis labels không bị cắt
   - ✅ X-axis chỉ ở subplot cuối cùng
   - ✅ Tất cả subplots share X-axis (zoom 1 cái → tất cả zoom)

## So sánh Overlay vs Stack

### Overlay mode:
```
┌────────────────────────────────────┐
│  All channels on same plot         │
│  ▬▬▬ Ch1  ▬▬▬ Ch2  ▬▬▬ Ch3         │
│                                    │
└────────────────────────────────────┘
```

### Stack mode:
```
┌────────────────────────────────────┐
│  Channel 1                         │
│  ▬▬▬▬▬▬▬▬▬▬▬▬                       │
├────────────────────────────────────┤
│  Channel 2                         │
│  ▬▬▬▬▬▬▬▬▬▬▬▬                       │
├────────────────────────────────────┤
│  Channel 3                         │
│  ▬▬▬▬▬▬▬▬▬▬▬▬                       │
└────────────────────────────────────┘
```

## Lưu ý

- **Height được tính động**: Số channels càng nhiều → height càng lớn
- **Minimum height 400px**: Ngay cả với 1 channel
- **Scrollbar tự động**: Nếu total height > container height
- **Responsive width**: Width vẫn fill container (từ fix trước)

---
Ngày fix: 2025-12-25
