# FIXED: Plots chỉ chiếm 50% width

## Vấn đề đã tìm ra

Plots chỉ hiển thị 50% width vì `#plot` div có CSS `display: flex` với `justify-content: center`.

## Root Cause

Trong `src/gui/widgets/plotly_view.py`, HTML template có:

```css
#plot {
  width: 100%;
  height: 100%;
  display: flex;           /* ← VẤN ĐỀ! */
  align-items: center;     /* ← VẤN ĐỀ! */
  justify-content: center; /* ← VẤN ĐỀ! */
}
```

**Tại sao gây lỗi:**
- `display: flex` + `justify-content: center` được dùng để center text "Loading Plotly..."
- Khi Plotly render plot, nó tạo child div bên trong `#plot`
- Flex container với `justify-content: center` làm cho child div chỉ chiếm content width
- Thay vì fill 100% width, plot chỉ chiếm khoảng 50% và được center

## Giải pháp

**Xóa flex properties khỏi #plot div:**

```css
#plot {
  width: 100%;
  height: 100%;
  background-color: #f5f5f5;
  /* Removed: display: flex, align-items, justify-content */
}
```

Text "Loading Plotly..." và error messages vẫn được center bằng inline style `text-align:center`.

## Files đã sửa

- `src/gui/widgets/plotly_view.py`:
  - Xóa `display: flex`, `align-items`, `justify-content` khỏi #plot CSS
  - Clean up debug logging

## Cách hoạt động

### Trước fix:
```
┌────────────────────────────────┐
│                                │
│     ████████████████████       │  ← Plot centered, chỉ 50% width
│                                │
└────────────────────────────────┘
```

### Sau fix:
```
┌────────────────────────────────┐
│ ██████████████████████████████ │  ← Plot fills 100% width
│                                │
└────────────────────────────────┘
```

## Test

```bash
source venv/bin/activate
python run_app.py
```

**Kết quả mong đợi:**
- ✅ Plots fill toàn bộ width của container
- ✅ Resize window → plots tự động resize theo
- ✅ Switch tabs → plots vẫn fill width
- ✅ Overlay và Stack modes đều fill width

## Technical Details

**CSS Flexbox behavior:**
- `display: flex` makes container a flex container
- `justify-content: center` centers flex items along main axis
- Flex items (child divs) size to their content by default
- This conflicts with Plotly's `autosize: true` which expects parent to provide full width

**Fix approach:**
- Remove flex from #plot div
- Let Plotly manage its own sizing with `autosize: true`
- Plotly will fill 100% of parent (#plot div)
- #plot div fills 100% of QWebEngineView
- QWebEngineView has `QSizePolicy.Expanding`

---
Ngày fix: 2025-12-25
Root cause: CSS `display: flex` + `justify-content: center`
Solution: Remove flex properties, let Plotly handle sizing
