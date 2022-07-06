hostname = '192.168.16.140'
lidarport = 7502
imuport = 7503

def pause_system():
    ccc = input("Press enter to continue")

from ouster import client
from datetime import datetime
import ouster.pcap as pcap
    
def config_sensor_once():    
    from ouster import client
    config = client.SensorConfig()
    config.lidar_mode = client.lidarMode.MODE_1024x10
    config.udp_port_lidar = 7502
    config.udp_port_imu = 7503
    config.operating_mode = client.OperatingMode.OPERATING_NORMAL

    #dual return on rev6 chips (OS1 has it), uncomment to use
    #config.udp_profile_lidar = client.UDPProfileLidar.PROFILE_LIDAR_FNG19_RFL8_SIG16_NIR16_DUAL
    
    #print("Configuration setup done")

    client.set_config(hostname,config,persist=True,udp_dest_auto=True)


#collect metadata
def collect_metadata():
    with open(metadata_path,'r') as f:
        metadata = client.SensorInfo(f.read())

    source = pcap.Pcap(pcap_path,metadata)

    for packet in source:
        if isinstance(packet,client.lidarPacket):
            #process lidarpackt, and access the measurement ids, timestamps and ranges
            measurement_ids = packet.measurement_id
            timestamps = packet.timestamp
            ranges = packet.field(client.ChanField.RANGE)
            #print(f' encoder counts = {measurement_ids.shape}')
            #print(f' timestamps = {timestamps.shape}')
            #print(f' ranges = {ranges.shape}')

        elif isinstance(packet,client.ImuPacket):
            #access imupacket
            #print(f' acceleration = {packet.accel}')
            #print(f' angular_velocity={packet.angular_vel}')
    

def scanner():
    #initialize lidar scanner
    h=640
    w=480
    #need to include something for the metadata var otherwise I can't use this
    ls=client.LidarScan(h,w,metadata.format.udp_profile_lidar)
    device = client.Sensor(hostname)
    return device
    
from more_itertools import time_limited
from contextlib import closing

n_seconds = 90

#connect to sensor and record lidar packs
with closing(client.Sensor(hostname,lidarport,imuport,buf_size=640)) as source:
    #make a descriptive filename for the metadata/pcap files

    time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = source.metadata

    fname_base = f"/media/ssd/{meta.prod_line}_{meta.sn}_{meta.mode}_{time_part}"
    
    #print(f"Saving sensor metadata to: {fname_base}.json")
    source.write_metadata(f"{fname_base}.json")

    #print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
    source_it = time_limited(n_seconds, source)
    n_packets = pcap.record(source_it, f"{fname_base}.pcap")

    #print(f"Captured {n_packets} packets")
