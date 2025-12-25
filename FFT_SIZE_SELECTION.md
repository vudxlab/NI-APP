# Hướng Dẫn Chọn Window Size FFT

## Tại sao có 2 cách chọn window size?

Ứng dụng có **2 chế độ FFT** với mục đích khác nhau:

### 1. **Real-time Mode** (Mặc định)
- **Mục đích**: Giám sát phổ tần liên tục
- **Window sizes**: Nhỏ (10ms - 640ms)
- **Chọn qua**: Dropdown **"FFT Size"**
- **Tự động cập nhật**: Có

### 2. **Long-Window Mode** (High-Resolution)
- **Mục đích**: Phân tích tần số thấp chi tiết
- **Window sizes**: Lớn (10s - 200s)
- **Chọn qua**: Dropdown **"Window"** (sau khi Save Buffer)
- **Tự động cập nhật**: Không (phải nhấn "Analyze")

---

## Cách Sử Dụng

### Mode Real-time

```
┌────────────────────────────────────────────────────────┐
│ FFT Mode: [Real-time ▼]                               │
│                                                        │
│ Scale: [dB ▼]  Freq Range: [Full ▼]  ...             │
│ FFT Size: [80 ms → 2048 pts (0.08 s, Δf=12.50 Hz) ▼] │  ← Thay đổi ở đây
│                                                        │
│ Plot cập nhật liên tục                                │
└────────────────────────────────────────────────────────┘
```

**Khi chọn FFT Size:**
- Chọn **10ms, 20ms, 50ms** → Cập nhật nhanh, độ phân giải thấp
- Chọn **80ms** (mặc định) → Cân bằng tốc độ/độ phân giải
- Chọn **200ms, 320ms, 640ms** → Cập nhật chậm hơn, độ phân giải cao hơn

**Bảng Window Sizes cho Real-time:**

| Time Window | FFT Size | Freq Resolution @ 25.6 kHz |
|-------------|----------|----------------------------|
| 10 ms | 256 | 100.0 Hz |
| 20 ms | 512 | 50.0 Hz |
| 50 ms | 1,024 | 25.0 Hz |
| **80 ms** ⭐ | **2,048** | **12.5 Hz** (Default) |
| 100 ms | 2,048 | 12.5 Hz |
| 200 ms | 4,096 | 6.25 Hz |
| 320 ms | 8,192 | 3.13 Hz |
| 640 ms | 16,384 | 1.56 Hz |

---

### Mode Long-Window

```
┌────────────────────────────────────────────────────────────────────────────┐
│ FFT Mode: [Long-Window (High-Res) ▼]                                      │
│                                                                            │
│ Save: [200s ▼] [Save Buffer] ✓ 5M samples (200.0s)                       │
│ Window: [200s ▼] [Analyze]  ← Thay đổi ở đây                             │
│                                                                            │
│ Scale: [dB ▼]  Freq Range: [0-20 Hz ▼]  ...                              │
│ (FFT Size bị ẩn vì không dùng)                                            │
│                                                                            │
│ Plot chỉ cập nhật khi nhấn "Analyze"                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

**Khi chọn Window:**
- Chọn **10s, 20s** → Nhanh, độ phân giải khá
- Chọn **50s, 100s** → Chậm hơn, độ phân giải cao
- Chọn **200s** (mặc định) → Chậm nhất, độ phân giải cao nhất

**Bảng Window Sizes cho Long-Window:**

| Time Window | FFT Size | Freq Resolution @ 25.6 kHz |
|-------------|----------|----------------------------|
| 10 s | 262,144 (2¹⁸) | 0.0977 Hz |
| 20 s | 524,288 (2¹⁹) | 0.0488 Hz |
| 50 s | 1,048,576 (2²⁰) | 0.0244 Hz |
| 100 s | 2,097,152 (2²¹) | 0.0122 Hz |
| **200 s** ⭐ | **4,194,304 (2²²)** | **0.0061 Hz** (Best) |

---

## So Sánh

| | Real-time Mode | Long-Window Mode |
|---|---|---|
| **Control** | "FFT Size" dropdown | "Window" dropdown |
| **Window** | 10ms - 640ms | 10s - 200s |
| **Độ phân giải** | 1.56 Hz - 100 Hz | 0.0061 Hz - 0.0977 Hz |
| **Cập nhật** | Tự động (100ms) | Thủ công (click "Analyze") |
| **Dùng cho** | Giám sát liên tục | Phân tích chi tiết |
| **Lưu file** | Không | Có (file tạm HDF5) |

---

## Câu Hỏi Thường Gặp

**Q: Tại sao không thấy dropdown "FFT Size" khi chọn Long-Window mode?**

A: Trong Long-Window mode, window size được chọn qua dropdown **"Window"** (10s-200s), không dùng "FFT Size" nữa. Đây là thiết kế có chủ đích để tránh nhầm lẫn.

**Q: Tại sao không thấy thay đổi khi chọn window size lớn hơn trong Real-time mode?**

A: Nếu bạn thấy các option như "10s, 20s, 200s" trong Real-time mode, đó là **bug của phiên bản cũ**. Phiên bản mới đã sửa:
- **Real-time**: Chỉ hiển thị 10ms-640ms
- **Long-Window**: Chỉ hiển thị 10s-200s

**Q: Window size nào tốt nhất?**

A: Tùy mục đích:
- **Giám sát real-time**: 80ms (mặc định)
- **Phát hiện tần số thấp (< 1 Hz)**: 200s
- **Cân bằng**: 100s

**Q: Có thể dùng window 200s trong Real-time mode không?**

A: Không. Window 200s yêu cầu 5.1M samples và tính toán quá lâu (>10s), không phù hợp cho real-time. Phải dùng Long-Window mode.

---

## Tips

### Tip 1: Chọn window phù hợp với tần số quan tâm

**Quy tắc:** Để phát hiện tần số `f`, cần window ≥ `1/f` giây

Ví dụ:
- Phát hiện 0.5 Hz → Cần window ≥ 2s → Dùng Long-Window mode
- Phát hiện 10 Hz → Cần window ≥ 0.1s → Dùng Real-time mode (100ms)

### Tip 2: Độ phân giải = Sample_Rate / Window_Size

Để phân biệt 2 tần số, độ phân giải phải nhỏ hơn khoảng cách giữa chúng.

Ví dụ: Phân biệt 0.5 Hz và 0.52 Hz
- Khoảng cách: 0.02 Hz
- Cần độ phân giải < 0.02 Hz
- → Cần window 200s (freq_res = 0.0061 Hz) ✅

### Tip 3: Chuyển đổi linh hoạt giữa 2 modes

1. Dùng Real-time để giám sát
2. Khi thấy có vấn đề → Chuyển Long-Window
3. Save 200s → Analyze với window 200s
4. Xem chi tiết → Quay lại Real-time

---

**Version:** 2.1 (2024-12-24)
