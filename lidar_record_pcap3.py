hostname = '192.168.16.140'
lidarport = 7502
imuport = 7503

def pause_system():
    ccc = input("Press enter to continue")

from ouster import client
from datetime import datetime
import ouster.pcap as pcap
from more_itertools import time_limited
from contextlib import closing
import argparse
import os

n_seconds = 300

a = argparse.ArgumentParser()
a.add_argument("out_path", help="path-to-output-folder")
args = a.parse_args()

#connect to sensor and record lidar packs
with closing(client.Sensor(hostname,lidarport,imuport,buf_size=640)) as source:

    #make a descriptive filename for the metadata/pcap files
    time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = source.metadata

    fname_base = os.path.join(args.out_path,f"{meta.prod_line}_{meta.sn}_{meta.mode}_{time_part}")
    
    #print(f"Saving sensor metadata to: {fname_base}.json")
    source.write_metadata(f"{fname_base}.json")

    #print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
    source_it = time_limited(n_seconds, source)
    n_packets = pcap.record(source_it, f"{fname_base}.pcap")

    #print(f"Captured {n_packets} packets")
