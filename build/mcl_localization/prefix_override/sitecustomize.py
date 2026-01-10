import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/felix/Schreibtisch/projects/robotics_hw2/install/mcl_localization'
