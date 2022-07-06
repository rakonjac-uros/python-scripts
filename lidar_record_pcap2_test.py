hostname = '192.168.16.140'
lidarport = 7502
imuport = 7503

def pause_system():
    ccc = input("Press enter to continue")


from datetime import datetime

import argparse
import os

n_seconds = 300

a = argparse.ArgumentParser()
a.add_argument("out_path", help="path-to-output-folder/")
args = a.parse_args()
#connect to sensor and record lidar packs

#make a descriptive filename for the metadata/pcap files
time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
fname_base = os.path.join(args.out_path,f"{time_part}")

if args.out_path == None:
    print("No path")
else:
    print("Path:", args.out_path)
json_out = f"{fname_base}.json"
print(f"Saving sensor metadata to: {json_out}")

pcap_out = f"{fname_base}.pcap"
#print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
print(f"Saving sensor data to: {pcap_out}")

#print(f"Captured {n_packets} packets")
