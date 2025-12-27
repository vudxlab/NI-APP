"""
Runtime hook for nidaqmx to find NI-DAQmx DLLs
"""
import os
import sys

# Add NI-DAQmx DLL paths to PATH (64-bit first, then 32-bit fallback)
ni_paths = [
    r"C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\lib64\msvc",
    r"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\lib32\msvc",
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
]

print("[HOOK] NI-DAQmx runtime hook starting...")

# Add to PATH
current_path = os.environ.get('PATH', '')
for ni_path in ni_paths:
    if os.path.exists(ni_path):
        os.environ['PATH'] = ni_path + os.pathsep + current_path
        print(f"[HOOK] Added to PATH: {ni_path}")

# Also add to DLL search path for Windows
if sys.platform == 'win32':
    try:
        for ni_path in ni_paths:
            if os.path.exists(ni_path):
                os.add_dll_directory(ni_path)
                print(f"[HOOK] Added DLL directory: {ni_path}")
    except (OSError, AttributeError) as e:
        print(f"[HOOK] Warning: Could not add DLL directory: {e}")

print("[HOOK] NI-DAQmx runtime hook completed.")
