from pymem import *
from pymem.process import *

def memoryValues():
    # 0x71 - Death Flag - 0
    # 0xF34 - Score - 1
    # 0x94 - Mario X Position - 2
    # 0xDBF - Coins - 4
    offsets = [0x71, 0xF34, 0x94, 0xDBF]
    pm = Pymem('snes9x-x64.exe')
    def GetPointer(offsets):
        values = []
        addr = pm.read_int(0x1408D8C40) 
        for offset in offsets:
            if(offset == 0x94):
                values.append(pm.read_bytes((addr + offset), 2))
            if(offset == 0xDBF):
                values.append(pm.read_bytes((addr + offset), 1))
            else:
                values.append(pm.read_int(addr + offset))
        return values
    values = (GetPointer(offsets))
    values[2] = int.from_bytes(values[2], "little")
    values[4] = int.from_bytes(values[4], "little")
    return values
    
def memoryReset():
    offsets = [0xF34]
    pm = Pymem('snes9x-x64.exe')
    addr = pm.read_int(0x1408D8C40) 
    for offset in offsets:
        pm.write_int((addr + offset), 0)

if __name__ == '__main__':
    print(memoryValues())
