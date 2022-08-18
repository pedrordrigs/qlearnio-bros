from pymem import *
from pymem.process import *

def memoryValues():
    offsets = [0xF34]

    pm = Pymem('snes9x-x64.exe')

    gameModule = module_from_name(pm.process_handle, 'snes9x-x64.exe').lpBaseOfDll

    def GetPointer(base, offsets):
        addr = pm.read_longlong(0x1408D8C40) 
        for offset in offsets:
            addr = pm.read_longlong(addr + offset)
        return addr

    score = (GetPointer(gameModule, offsets))*10
    return score
memoryValues()