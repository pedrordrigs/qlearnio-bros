from directkeys import PressKey, ReleaseKey
from inputcodes import RUN, JUMP, SPIN, LEFT, RIGHT, RESET
import time

while(1):
    PressKey(RIGHT)
    time.sleep(3)
    ReleaseKey(RIGHT)