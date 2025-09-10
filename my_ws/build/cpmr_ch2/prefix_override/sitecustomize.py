import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/common/Documents/argh/my_ws/install/cpmr_ch2'
