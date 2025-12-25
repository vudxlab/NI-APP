# Hướng Dẫn Sử Dụng Long-Window FFT (Tần Số Thấp)

## Giới thiệu

Chức năng **Long-Window FFT** được tích hợp vào tab **"Frequency Domain"** để phân tích **tần số thấp** với độ phân giải cao.

### Tại sao cần Long-Window FFT?

- **FFT thông thường (Real-time)**: Window size nhỏ (2048 samples @ 25.6 kHz = 0.08s) → độ phân giải thấp (~12.5 Hz)
- **Long-Window FFT**: Window size lớn (5,120,000 samples @ 25.6 kHz = 200s) → độ phân giải cao (~0.0061 Hz)

**Ứng dụng:**
- Phát hiện rung động tần số thấp (< 1 Hz)
- Giám sát kết cấu công trình (structural monitoring)
- Phân tích hiện tượng biến đổi chậm
- Phát hiện các mode rung động kết cấu

---

## Cách Sử Dụng

### Bước 1: Khởi động và thu dữ liệu

1. Mở ứng dụng NI DAQ Vibration Analysis
2. Cấu hình DAQ (tab "DAQ")
3. Nhấn **Start** để bắt đầu thu dữ liệu
4. Chờ ít nhất **50-200 giây** để buffer có đủ dữ liệu

### Bước 2: Chuyển sang tab "Frequency Domain"

Click vào tab **"Frequency Domain"** ở phần Plot tabs

### Bước 3: Chuyển sang chế độ Long-Window FFT

**Quan trọng:** Có 2 chế độ FFT:

#### 🔹 **Real-time** (Mặc định)
- FFT real-time với window ngắn
- Cập nhật liên tục
- Độ phân giải thấp (~12.5 Hz)

#### 🔹 **Long-Window (High-Res)** ⭐ Chế độ phân tích tần số thấp
- FFT với window dài (10s-200s)
- Lưu dữ liệu vào file tạm
- Độ phân giải cao (~0.0061 Hz với window 200s)

**→ Chọn "Long-Window (High-Res)" từ dropdown "FFT Mode"**

### Bước 4: Lưu dữ liệu buffer

Khi chọn mode "Long-Window", sẽ hiện ra controls:

```
FFT Mode: [Long-Window (High-Res) ▼]  Save: [200s ▼] [Save Buffer] ✓ 5120000 samples (200.0s)  Window: [200s ▼] [Analyze]
```

1. **Chọn "Save"** duration:
   - **50s**: Phân tích nhanh
   - **100s**: Cân bằng
   - **200s**: Độ phân giải cao nhất ⭐

2. **Nhấn "Save Buffer"**:
   - Dữ liệu được lưu vào `/tmp/ni_app_long_data/`
   - Status hiển thị số samples và duration
   - Ví dụ: `✓ 5120000 samples (200.0s)`

3. **Nếu lỗi "No data buffer available"**:
   - Đảm bảo đã Start acquisition
   - Chờ ít nhất 200s để buffer đủ dữ liệu

### Bước 5: Chọn cửa sổ FFT và phân tích

**LƯU Ý QUAN TRỌNG:**
- Trong **Real-time mode**: Dùng dropdown **"FFT Size"** (hiển thị 10ms-640ms)
- Trong **Long-Window mode**: Dùng dropdown **"Window"** (hiển thị 10s-200s)
- Dropdown "FFT Size" sẽ **tự động ẩn** khi chuyển sang Long-Window mode!

1. **Chọn "Window"** (chỉ hiện trong Long-Window mode):
   - **10s** → freq_res = 0.0977 Hz
   - **20s** → freq_res = 0.0488 Hz
   - **50s** → freq_res = 0.0244 Hz
   - **100s** → freq_res = 0.0122 Hz
   - **200s** → freq_res = 0.0061 Hz ⭐ **Khuyến nghị cho tần số thấp**

2. **(Tùy chọn) Chọn "Freq Range"** để zoom vào dải tần quan tâm:
   - **0-20 Hz**: Rất thấp
   - **0-100 Hz**: Thấp-trung bình
   - **Full**: Toàn bộ phổ

3. **Nhấn "Analyze"**:
   - FFT được tính toán cho tất cả channels đang hiển thị
   - Kết quả hiển thị trên plot giống Real-time mode
   - Peaks được detect và hiển thị (nếu bật "Show Peaks")

### Bước 6: Xem kết quả

Kết quả hiển thị trên **cùng một plot** như Real-time mode:
- **Biểu đồ**: Phổ tần số với độ phân giải cao
- **Peaks**: Chấm đỏ đánh dấu các đỉnh
- **Zoom**: Có thể zoom vào các khu vực quan tâm
- **Channels**: Có thể ẩn/hiện từng channel

---

## Bảng So Sánh 2 Chế Độ

| Tính năng | Real-time | Long-Window (High-Res) |
|-----------|-----------|------------------------|
| **Cập nhật** | Liên tục (100ms) | Thủ công (nhấn Analyze) |
| **Window size** | 256-16384 samples | 262,144 - 4,194,304 samples |
| **Duration** | 0.01s - 0.64s | 10s - 200s |
| **Freq Resolution** | ~12.5 Hz | 0.0061 Hz - 0.0977 Hz |
| **Dùng cho** | Giám sát real-time | Phân tích tần số thấp chi tiết |
| **Lưu file tạm** | ❌ Không | ✅ Có |
| **Tốc độ** | Nhanh | Chậm hơn (vài giây) |

---

## Quy Trình Làm Việc Thực Tế

### Scenario 1: Giám sát liên tục + Phân tích chi tiết khi cần

1. Để ở mode **Real-time** để xem overview
2. Khi thấy có điều bất thường:
   - Chuyển sang **Long-Window**
   - Save buffer 200s
   - Analyze với window 200s
   - Set Freq Range 0-20 Hz để zoom vào dải thấp
3. Sau khi phân tích xong, quay lại **Real-time**

### Scenario 2: Phân tích chuyên sâu ngay từ đầu

1. Start acquisition, chờ 200s
2. Chuyển sang **Long-Window**
3. Save 200s → Analyze với window 200s
4. Dùng "Show Peaks" để tìm tần số nổi bật
5. Điều chỉnh "Freq Range" để zoom vào khu vực quan tâm
6. Lặp lại với các window khác (50s, 100s) để so sánh

---

## Độ Phân Giải Tần Số

@ Sample Rate = 25.6 kHz:

| Window | Samples (2^N) | Freq Resolution | Có thể phân biệt |
|--------|---------------|-----------------|------------------|
| **Real-time** | | | |
| 0.01s | 256 | 100.0 Hz | Rất thô |
| 0.08s | 2048 | 12.5 Hz | Thô |
| **Long-Window** | | | |
| 10s | 262,144 (2¹⁸) | 0.0977 Hz | f1=0.5 Hz vs f2=0.6 Hz ✅ |
| 20s | 524,288 (2¹⁹) | 0.0488 Hz | f1=0.5 Hz vs f2=0.55 Hz ✅ |
| 50s | 1,048,576 (2²⁰) | 0.0244 Hz | f1=0.5 Hz vs f2=0.53 Hz ✅ |
| 100s | 2,097,152 (2²¹) | 0.0122 Hz | f1=0.5 Hz vs f2=0.52 Hz ✅ |
| 200s | 4,194,304 (2²²) | 0.0061 Hz | f1=0.5 Hz vs f2=0.51 Hz ✅ |

**Quy tắc**: `|f2 - f1| > Frequency Resolution` để phân biệt được

---

## Tips & Tricks

### 1. Khi nào dùng Long-Window?

✅ **Nên dùng khi:**
- Phát hiện tần số < 1 Hz
- Cần phân biệt 2 tần số gần nhau (< 0.5 Hz)
- Phân tích structural modes
- Nghiên cứu chi tiết phổ tần

❌ **Không cần dùng khi:**
- Giám sát real-time
- Phát hiện tần số > 10 Hz
- Cần update nhanh

### 2. Tối ưu hóa

- **File tạm**: Tự động xóa sau 24 giờ
- **Memory**: Save 200s @ 4 ch ≈ 156 MB RAM
- **CPU**: Tắt real-time update khi analyze long-window để giảm tải

### 3. Phân tích đa channel

- Có thể ẩn/hiện channels bằng legend
- Long FFT phân tích tất cả channels hiển thị cùng lúc
- Kết quả overlay trên cùng 1 plot

### 4. Xử lý lỗi thường gặp

**"No data buffer available"**
→ Start acquisition trước khi Save Buffer

**"Insufficient Data"**
→ Đợi lâu hơn hoặc chọn duration/window nhỏ hơn

**"Not Initialized"**
→ Khởi động lại app hoặc re-configure DAQ

---

## Keyboard Shortcuts (Coming Soon)

- `Ctrl+L`: Toggle Long-Window mode
- `Ctrl+S`: Save Buffer
- `Ctrl+A`: Analyze
- `Ctrl+R`: Switch to Real-time mode

---

## So sánh với Tab riêng (Phiên bản cũ)

Phiên bản trước có tab riêng "Long-Window FFT". Bây giờ đã **gộp vào tab "Frequency Domain"** với ưu điểm:

✅ **Giao diện gọn hơn**: 1 tab thay vì 2
✅ **Linh hoạt hơn**: Chuyển đổi nhanh Real-time ↔ Long-Window
✅ **Ít lặp lại**: Dùng chung plot, controls, settings
✅ **Hiệu quả hơn**: Không cần chuyển tab

---

## Technical Details

### File Format

**Temporary Files:**
- Location: `/tmp/ni_app_long_data/`
- Format: HDF5 (`.h5`)
- Compression: GZIP level 4
- Auto-cleanup: 24 hours

### Memory Usage

| Duration | Channels | RAM | Disk (HDF5) |
|----------|----------|-----|-------------|
| 50s | 4 | ~39 MB | ~24 MB |
| 100s | 4 | ~78 MB | ~47 MB |
| 200s | 4 | ~156 MB | ~94 MB |

### Processing Time

@ Intel i5 / 16GB RAM:

| Window | Total Time |
|--------|-----------|
| 10s | ~1s |
| 20s | ~2s |
| 50s | ~4s |
| 100s | ~9s |
| 200s | ~18s |

---

## FAQ

**Q: Tại sao phải lưu vào file thay vì dùng trực tiếp từ buffer?**

A:
1. FFT với window lớn tốn nhiều RAM
2. Lưu file cho phép phân tích lại nhiều lần
3. HDF5 compression tiết kiệm 40% dung lượng

**Q: Có thể phân tích nhiều window cùng lúc không?**

A: Không, phải chọn từng window một. Nhưng có thể:
1. Analyze với 10s → Screenshot
2. Analyze với 200s → Screenshot
3. So sánh kết quả

**Q: Tại sao không thấy peaks trong plot?**

A:
1. Tắt "Show Peaks" → Bật lại
2. Threshold quá cao → Giảm threshold
3. Không có tần số nổi bật → Zoom vào dải thấp (0-20 Hz)

**Q: File tạm có bị mất không?**

A: Có, sau 24h hoặc khi restart máy. Nếu cần giữ lâu dài, nên export sang CSV hoặc copy file HDF5 ra ngoài.

**Q: Có thể dùng Real-time và Long-Window đồng thời không?**

A: Không, phải chọn 1 trong 2 mode. Khuyến nghị:
- **Real-time**: Giám sát liên tục
- **Long-Window**: Khi cần phân tích chi tiết

---

## Version History

- **v2.0** (2024-12-24): **Gộp vào Frequency Domain tab**
  - Tích hợp controls vào FFT plot widget
  - Xóa tab riêng "Long-Window FFT"
  - Toggle giữa Real-time và Long-Window mode
  - Giữ nguyên giao diện plot hiện có

- **v1.0** (2024-12-24): Initial release
  - Tab riêng "Long-Window FFT"
  - 5 window durations: 10s-200s
  - HDF5 file format

---

**© 2024 NI-APP Vibration Analysis System**
