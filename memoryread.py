from pymem import *
from pymem.process import *
const DEATH_FLAG = 0x71
const SCORE = 0xF34
const X_POSITION = 0x94
const COINS = 0xDBF
const MEMORY_ADDRESS = 0x1408D8C40
def memoryValues():
    # 0x71 - Death Flag - 0
    # 0xF34 - Score - 1
    # 0x94 - Mario X Position - 2
    # 0xDBF - Coins - 4
    offsets = [DEATH_FLAG, SCORE, X_POSITION, COINS]
    pm = Pymem('snes9x-x64.exe')
    def GetPointer(offsets):
        values = []
        addr = pm.read_int(MEMORY_ADDRESS) 
        for offset in offsets:
            if(offset == X_POSITION):
                values.append(pm.read_bytes((addr + offset), 2))
            if(offset == COINS):
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
    addr = pm.read_int(MEMORY_ADDRESS) 
    for offset in offsets:
        pm.write_int((addr + offset), 0)

if __name__ == '__main__':
    print(memoryValues())
