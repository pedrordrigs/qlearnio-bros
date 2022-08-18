from ctypes import *
from ctypes.wintypes import *

OpenProcess = windll.kernel32.OpenProcess
ReadProcessMemory = windll.kernel32.ReadProcessMemory
CloseHandle = windll.kernel32.CloseHandle

PROCESS_ALL_ACCESS = 0x1F0FFF

pid = 1672   # I assume you have this from somewhere.
address = 0x96  # Likewise; for illustration I'll get the .exe header.

buffer = c_wchar_p("The data goes here")
bufferSize = len(buffer.value)
bytesRead = c_ulong(0)

processHandle = OpenProcess(PROCESS_ALL_ACCESS, False, pid)
if ReadProcessMemory(processHandle, address, buffer, bufferSize, byref(bytesRead)):
    print ("Success:", buffer)
else:
    print("Failed.")
CloseHandle(processHandle)