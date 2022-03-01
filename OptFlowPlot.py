# -*- coding: utf-8 -*-
"""
Extract and plot NoGPS vs GPS data
"""
import csv
import numpy as np
from pyproj import Proj
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from mpl_toolkits import mplot3d
import pymap3d as pm

plot_base = 0
plot_rot = 0
plot_rot_scale = 0
plot_imu = 0
plot_master = 0
plot_data = 1

dim2 = True
data_set = 9


# Crtati grafike za razlicite kombinacije ovih parametara
sample_size = 20
rotation_burn = 5
scale_burn = 10


pr = Proj(proj='utm', zone = 34, ellps='WGS84',preserve_units=False)


if data_set == 1:
    dir_path = "/home/uros/NoGPSdata2/"
    csv_path = dir_path + "noGPSdata/set2.csv" # Front facing (set 2)
    traj_path = dir_path + "noGPSimages/KeyFrameTrajectory_front_facing.txt" # Front facing 1 
    times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2
    first_frame_ts = '20210802_162610_347' # Front facing (set 2)
    first_pose_ts  = '20210802_162619_789' # Front facing 1
    first_pose_frame = 96 # Front facing 1
    v0_frame_ts    = '20210802_162612_433' # Front facing 1
    dim2 = True
elif data_set == 2:
    dir_path = "/home/uros/NoGPSdata2/"
    csv_path = dir_path + "noGPSdata/set2.csv" # Front facing (set 2)
    traj_path = dir_path + "noGPSimages/KeyFrameTrajectory_FF_20_7.txt" # Front facing 2 (FF_20_7)
    times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2
    first_frame_ts = '20210802_162610_347' # Front facing (set 2)
    first_pose_ts = '20210802_162614_088' # Front facing 2 (FF_20_7)
    first_pose_frame = 45 # Front facing 2 (FF_20_7)
    v0_frame_ts = '20210802_162614_088' # Front facing 2 (FF_20_7) - Not working
    dim2 = True
elif data_set == 3:
    dir_path = "/home/uros/NoGPSdata2/"
    csv_path = dir_path + "noGPSdata/set2.csv" # Front facing (set 2)
    traj_path = dir_path + 'result_short.txt' # DSO set 2 short
    times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2
    first_frame_ts = '20210802_162558_741' # DSO Front facing (set 2)
    first_pose_ts = '20210802_162634_793' # DSO Front facing (set 2) short
    first_pose_frame = 388 # DSO Front facing (set 2) short
    v0_frame_ts    = '20210802_162634_793' # Down facing - Not working
    dim2 = True
elif data_set == 4:
    dir_path = "/home/uros/NoGPSdata2/"
    csv_path = dir_path + "noGPSdata/set2.csv" # Front facing (set 2)
    traj_path = dir_path + "noGPSimages/KeyFrameTrajectory_FF_20_7_part2.txt" # Front facing 2 after first turn (FF_20_7_part2)
    times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2
    first_frame_ts = '20210802_162610_347' # Front facing (set 2)
    first_pose_ts = '20210802_162659_749' # Front facing 2 after first turn (FF_20_7_part2)
    first_pose_frame = 320 # Front facing 2 after first turn (FF_20_7_part2)
    v0_frame_ts    = '20210802_162614_088' # Front facing 2 after first turn (FF_20_7_part2) - Not working
    dim2 = True
elif data_set == 5:
    dir_path = "/home/uros/NoGPSdata2/"
    csv_path = dir_path + "noGPSdata/set4.csv" # Down facing (set 4)
    traj_path = dir_path + "noGPSimages/KeyFrameTrajectory_grass_forward.txt" # Down facing (set 4) only forward    
    times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2
    first_frame_ts = '20210802_163807_661' # Down facing (set 4)
    first_pose_ts = '20210802_163814_936' # Down facing (set 4) only forward
    first_pose_frame = 143 # Down facing (set 4) only forward
    v0_frame_ts    = '20210802_163814_936' # Not working
    dim2 = True
elif data_set == 6:
    dir_path = "/home/uros/NoGPSdata3/"
    csv_path = dir_path + "f02data/30_20210803_125559.csv" # Down facing (fligth 02) 
    traj_path = dir_path + "KeyFrameTrajectory_f02short_firstForward.txt" # Down facing (fligth 02 short) turn home 
    times_path = dir_path + "times.txt" # NoGPSdata3
    first_frame_ts = '20210803_125609_509' # Down facing (fligth 02 short) first forward
    first_pose_ts = '20210803_125707_748' # Down facing (fligth 02 short) first forward
    first_pose_frame = 534 # Down facing (fligth 02 short) first forward
    v0_frame_ts    = '20210803_125707_748' # Not working
elif data_set == 7:
    dir_path = "/home/uros/NoGPSdata3/"
    csv_path = dir_path + "f02data/30_20210803_125559.csv" # Down facing (fligth 02) 
    traj_path = dir_path + "KeyFrameTrajectory_f02_turnHome.txt" # Down facing (fligth 02 short) turn home
    times_path = dir_path + "times.txt" # NoGPSdata3
    first_frame_ts = '20210803_125609_509' # Down facing (fligth 02 short) first forward
    first_pose_ts = '20210803_125824_678' # Down facing (fligth 02 short) turn home
    first_pose_frame = 1198 # Down facing (fligth 02 short) turn home
    v0_frame_ts    = '20210803_125824_678' # Not working
elif data_set == 8:
    dir_path = "/home/uros/NoGPSdata2/"
    csv_path = dir_path + "noGPSdata/set2.csv" # Front facing (set 2)
    traj_path = dir_path + 'result.txt' # DSO set 2
    times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2
    first_frame_ts = '20210802_162558_741' # DSO Front facing (set 2)
    first_pose_ts = '20210802_162611_781' # DSO Front facing (set 2)
    first_pose_frame = 220 # DSO Front facing (set 2)
    v0_frame_ts    = '20210802_162611_781' # Down facing - Not working
    
elif data_set == 9:
    dir_path = "/home/uros/NoGPSdata/uki2/"
    csv_path = dir_path + "data/set_uki2.csv" # Down facing (uki2)
    traj_path = dir_path + 'KeyFrameTrajectory1.txt' # uki2
    times_path = dir_path + "times.txt" # uki2
    first_frame_ts = '20210719_184524_543' # 
    first_pose_ts = '20210719_184534_254' # 
    first_pose_frame = 189 # 
    v0_frame_ts    = '20210719_184534_254' #

#dir_path = "/home/uros/NoGPSdata3/"


#csv_path = dir_path + "noGPSdata/set4.csv" # Down facing (set 4)

#csv_path = dir_path + "f02data/30_20210803_125559.csv" # Down facing (fligth 02) 




#traj_path = dir_path + "noGPSimages/KeyFrameTrajectory_grass_forward.txt" # Down facing (set 4) only forward

#traj_path = dir_path + "KeyFrameTrajectory_f02_turnHome.txt" # Down facing (fligth 02 short) first forward

#traj_path = dir_path + 'result.txt' # DSO set 2

#traj_path = dir_path + 'result_short.txt' # DSO set 2 short


#times_path = dir_path + "noGPSimages/times.txt" # NoGPSdata2

#times_path = dir_path + "times.txt" # NoGPSdata3


#first_frame_ts = '20210802_162610_347' # Front facing (set 2)

#first_frame_ts = '20210802_163807_661' # Down facing (set 4)

#first_frame_ts = '20210803_125609_509' # Down facing (fligth 02 short) first forward

#first_frame_ts = '20210802_162558_741' # DSO Front facing (set 2)

#first_pose_ts = '20210802_162614_088' # Front facing 2 (FF_20_7)
#first_pose_ts = '20210802_162659_749' # Front facing 2 after first turn (FF_20_7_part2)

#first_pose_ts = '20210802_163814_936' # Down facing (set 4) only forward

#first_pose_ts = '20210803_125707_748' # Down facing (fligth 02 short) first forward

#first_pose_ts = '20210803_125824_678' # Down facing (fligth 02 short) turn home

#first_pose_ts = '20210802_162611_781' # DSO Front facing (set 2)

#first_pose_ts = '20210802_162634_793' # DSO Front facing (set 2) short


#v0_frame_ts    = '20210802_162612_433' # Front facing 1
#v0_frame_ts    = '20210802_162614_088' # Front facing 2 (FF_20_7) - Not working
#v0_frame_ts    = '20210802_162614_088' # Front facing 2 after first turn (FF_20_7_part2) - Not working
#v0_frame_ts    = '20210802_163814_936' # Down facing - Not working

#v0_frame_ts    = '20210802_162634_793' # Down facing - Not working


#first_pose_frame = 96 # Front facing 1
#first_pose_frame = 45 # Front facing 2 (FF_20_7)
#first_pose_frame = 320 # Front facing 2 after first turn (FF_20_7_part2)

#first_pose_frame = 143 # Down facing (set 4) only forward

#first_pose_frame = 534 # Down facing (fligth 02 short) first forward
#first_pose_frame = 1198 # Down facing (fligth 02 short) turn home
#first_pose_frame = 220 # DSO Front facing (set 2)
#first_pose_frame = 388 # DSO Front facing (set 2) short



def lon2meters(x): # Degrees
    rad = x*np.pi/180.0
    return ((111415.13 * np.cos(rad))- (94.55 * np.cos(3.0*rad)) + (0.12 * np.cos(5.0*rad)))

def lat2meters(x): # Degrees
    rad = x*np.pi/180.0
    return(111132.09 - (566.05 * np.cos(2.0*rad))+ (1.20 * np.cos(4.0*rad)) - (0.002 * np.cos(6.0*rad)))


def rotationMatrix(x1,y1,x2,y2):
    R = np.zeros((2,2))
    R = [[x1*x2 + y1*y2, x2*y1 - x1*y2], 
         [x1*y2 - x2*y1, x1*x2 + y1*y2]]
    return R

def rotationMatrix2(x1,y1,x2,y2): # 1 to 2
    N1 = (x1**2 + y1**2)
    N2 = (x2**2 + y2**2)
    if (N1 == 0):
        N1 = 1
    if (N2 == 0):
        N2 = 1
    x1 = x1 / np.sqrt(N1)
    y1 = y1 / np.sqrt(N1)
    
    x2 = x2 / np.sqrt(N2)
    y2 = y2 / np.sqrt(N2)
    if ((x1**2 + y1**2)>0.9 and (x1**2 + y1**2)<1.1 and (x2**2 + y2**2)>0.9 and (x2**2 + y2**2)<1.1): 
        sint = (x2*y1 - x1*y2) / N1
        cost = (x2*x1 + y1*y2) / N1
            
        R = np.array([[cost, sint],[-sint, cost]])
        
    else:
        R = np.array([[0, 0],[0, 0]])
    return R

def rotationMatrix3(x1,y1,x2,y2,z1,z2): # 1 to 2
    RzB = rotationMatrix2(x1, y1, x2, y2)
    cs = RzB[0][0]
    sn = RzB[0][1]
    Rz = np.array([[cs, sn, 0],
                   [-sn,  cs, 0],
                   [0,   0,  1]])
                   
    RyB = rotationMatrix2(z1, x1, z2, x2)
    cs = RyB[0][0]
    sn = RyB[0][1]
    Ry = np.array([[cs , 0, -sn],
                   [0  , 1,  0],
                   [sn, 0, cs]])
    
    RxB = rotationMatrix2(y1, z1, y2, z2)
    cs = RxB[0][0]
    sn = RxB[0][1]
    Rx = np.array([[1 ,  0,   0],
                   [0 , cs, sn],
                   [0,  -sn,  cs]])
    
    return np.matmul(np.matmul(Rz, Ry), Rx)

def rotationMatrix3D(x1,y1,z1,x2,y2,z2):
    a = np.array([x1,y1,z1])
    b = np.array([x2,y2,z2])
    Na = np.sqrt(np.sum(a**2))
    Nb = np.sqrt(np.sum(b**2))
    Na = Na if Na != 0 else 1
    Nb = Nb if Nb != 0 else 1
    a = a / Na
    b = b / Nb
    v = np.cross(a,b)
    c = np.dot(a, b)
    vx = np.array([[0,    -v[2],  v[1]],
                   [v[2],     0, -v[0]],
                   [-v[1], v[0],    0]])
    R = np.eye(3) + vx + np.dot(vx,vx) / (1 + c)
    
    return R
    
    

def rotateCoordSys(x,y,R):
    X = np.ndarray((2,1))
    X[0] = x
    X[1] = y
    return np.matmul(R,X)

def rotateCoordSys2(x,y,R):
    N1 = (x**2 + y**2)
    if (N1 == 0):
        N1 = 1
    x = x / np.sqrt(N1)
    y = y / np.sqrt(N1)

    X = np.ndarray((2,1))
    X[0] = x
    X[1] = y
    Xr = np.matmul(R,X)
    Xr[0] = Xr[0] * np.sqrt(N1)
    Xr[1] = Xr[1] * np.sqrt(N1)
    return Xr

def rotateCoordSys3D(x,y,z,R):
    X = np.ndarray((3,1))
    X[0] = x
    X[1] = y
    X[2] = z
    return np.matmul(R,X)

def timestamp2ms(timestamp):
    ts = timestamp.split('_')
    h = int(ts[1][0:2])
    m = int(ts[1][2:4])
    s = int(ts[1][4:6])
    ms = int(ts[2])
    return ms + s*1000 + m*60*1000 + h*60*60*1000

csv_file = open(csv_path,"r") 

dict_reader = csv.DictReader(csv_file)

ordered_dict_from_csv = list(dict_reader)

data_list = []
for i in range(0,len(ordered_dict_from_csv)):
    data_list.append(dict(ordered_dict_from_csv[i]))

csv_file.close() 


times_ms_all = []
for i in range(0,len(data_list)):
    times_ms_all.append(timestamp2ms(data_list[i]['Timestamp']))
    
    
with open(traj_path, 'r') as f:
	traj_data = f.readlines()

f.close()

key_frames = []
x = []
y = []
z = []

for i, s in enumerate(traj_data):
    d = traj_data[i].split()
    key_frames.append(int(float(d[0])))
    x.append(float(d[1]))
    y.append(float(d[2]))
    z.append(float(d[3]))

x = -(np.array(x) - x[0])
y = np.array(y) - y[0]
z = np.array(z) - z[0]

with open(times_path, 'r') as f:
	frames = list(map(int, f.readlines()))

f.close()

start_idx = 0

for i in range(0,len(data_list)):
    if (data_list[i]['Timestamp'] == first_frame_ts):
        break;
    start_idx += 1
    
v0_idx = 0
for i in range(0,len(data_list)):
    if (data_list[i]['Timestamp'] == v0_frame_ts):
        break;
    v0_idx += 1
        

sensor_data = data_list[start_idx:].copy()
times_ms = times_ms_all[v0_idx:].copy()
imu_data = data_list[v0_idx:].copy()

imux = [0]
imuy = [0]
x0, y0 = pr(float(imu_data[0]['Latitude']), float(imu_data[0]['Longitude']))
xi = [0]
yi = [0]
v0x = 0
v0y = 0
axs = []
ays = []
vxs = [0]
vys = [0]
for i in range(1,len(times_ms)):
    #axt = float(imu_data[i-1]['Drone lin_acc x'])
    #ayt = float(imu_data[i-1]['Drone lin_acc y'])
    axt = float(imu_data[i-1]['Drone pitch'])
    ayt = float(imu_data[i-1]['Drone pitch'])
    axs.append(axt)
    ays.append(ayt)
    xt, yt = pr(float(imu_data[i]['Latitude']),imu_data[i]['Longitude'])
    xi.append(xt - x0)
    yi.append(yt - y0)

    dt = float(times_ms[i] - times_ms[i-1])/1000.0
    dx = v0x*dt + (axt*dt**2)/2
    dy = v0y*dt + (ayt*dt**2)/2
    v0x += axt*dt
    v0y += ayt*dt
    vxs.append(v0x)
    vys.append(v0y)
    imux.append(imux[i-1] + dx)
    imuy.append(imuy[i-1] + dy)
    

    
lat = []
lon = []
alt = []
x_ = []
y_ = []
z_ = []
ax = []
ay = []
az = []

cam_pitch = []
cam_yaw = []
drn_pitch = []
drn_roll = []
drn_yaw = []

cam_pitch2 = []
cam_yaw2 = []
drn_pitch2 = []
drn_roll2 = []
drn_yaw2 = []
#x0, y0 = pr(float(sensor_data[key_frames[0] - 1]['Latitude']), float(sensor_data[key_frames[0] - 1]['Longitude']))

lat0 = float(sensor_data[key_frames[0] - 1]['Latitude'])
lon0 = float(sensor_data[key_frames[0] - 1]['Longitude'])
h0 = float(sensor_data[key_frames[0] - 1]['Rel. altitude'])/1000.0
x0,y0,z0 = pm.geodetic2enu(lat0, lon0, h0, lat0, lon0, h0)
for i in range(0,len(key_frames)):
    lat.append(float(sensor_data[key_frames[i] - 1]['Latitude']))
    lon.append(float(sensor_data[key_frames[i] - 1]['Longitude']))
    alt.append(float(sensor_data[key_frames[i] - 1]['Rel. altitude'])/1.0)
    #xt, yt = pr(lat[i],lon[i])
    xt,yt,zt = pm.geodetic2enu(lat[i], lon[i],alt[i], lat0, lon0, h0)
    x_.append(xt - x0)
    y_.append(yt - y0)
    #z_.append(alt[i])
    z_.append(zt - h0)
    
# =============================================================================
#     ax.append(float(sensor_data[key_frames[i] - 1]['Drone lin_acc x']))
#     ay.append(float(sensor_data[key_frames[i] - 1]['Drone lin_acc y']))
#     az.append(float(sensor_data[key_frames[i] - 1]['Drone lin_acc z']))
# =============================================================================
    ax.append(float(sensor_data[key_frames[i] - 1]['Drone pitch']))
    ay.append(float(sensor_data[key_frames[i] - 1]['Drone pitch']))
    az.append(float(sensor_data[key_frames[i] - 1]['Drone pitch']))
    cam_pitch.append(float(sensor_data[key_frames[i] - 1]['Camera pitch']))
    cam_yaw.append(float(sensor_data[key_frames[i] - 1]['Camera yaw']))
    drn_pitch.append(float(sensor_data[key_frames[i] - 1]['Drone pitch']))
    drn_roll.append(float(sensor_data[key_frames[i] - 1]['Drone roll']))
    drn_yaw.append(float(sensor_data[key_frames[i] - 1]['Drone heading']))
    
for j in range(0,len(sensor_data)):    
    cam_pitch2.append(float(sensor_data[j]['Camera pitch']))
    cam_yaw2.append(float(sensor_data[j]['Camera yaw']))
    drn_pitch2.append(float(sensor_data[j]['Drone pitch']))
    drn_roll2.append(float(sensor_data[j]['Drone roll']))
    drn_yaw2.append(float(sensor_data[j]['Drone heading']))
        
# =============================================================================
# scales_x = []
# scales_y = []
# scales_z = []
# 
# for i in range(1,len(x)):
#     if((x_[i] - x_[i-1]) != 0 and (y_[i] - y_[i-1]) != 0):
#         scales_x.append((x_[i] - x_[i-1])/(x[i]-x[i-1]))
#         scales_y.append((y_[i] - y_[i-1])/(y[i]-y[i-1]))
#         scales_z.append((z_[i] - z_[i-1])/(z[i]-z[i-1]))
# =============================================================================
    
# Scale factor is inconsistant
# =============================================================================
# scale_x = np.mean(np.abs(scales_x[10:]))
# scale_y = np.mean(np.abs(scales_y[10:]))
# =============================================================================
    
    
if plot_base:
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('NoGPS')
    plt.show()
    
    plt.plot(x_, y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('GPS')
    plt.show()

Rs = []

if dim2:
    for i in range(0,len(x)):
        Rs.append(rotationMatrix2(x[i], y[i], x_[i], y_[i]))
       
    
    R = np.mean(Rs[rotation_burn:min(rotation_burn+sample_size, len(Rs))], axis = 0)
    
    xr = []
    yr = []
    zr = []
    for i in range(0,len(x)):
        X_t = rotateCoordSys(x[i],y[i],R)
        xr.append(X_t[0][0])
        yr.append(X_t[1][0])
        zr.append(0)
else:
# =============================================================================
#     for i in range(0,len(x)):
#         Rs.append(rotationMatrix3D(x[i], y[i], z[i], x[i], y[i], 0))
#        
#     
#     R = np.mean(Rs[rotation_burn:min(rotation_burn+sample_size, len(Rs))], axis = 0)
#     #Rs = []
#     
#     xrt = []
#     yrt = []
#     zrt = []
# 
#     for i in range(0,len(x)):
#         X_t = rotateCoordSys3D(x[i],y[i],z[i],Rs[i])
#         xrt.append(X_t[0][0])
#         yrt.append(X_t[1][0])
#         zrt.append(X_t[2][0])
# =============================================================================
    Rs = []   
    for i in range(0,len(x)-1):
        #Rs.append(rotationMatrix3D(xrt[i], yrt[i], zrt[i], x_[i], y_[i],0))
        Rs.append(rotationMatrix3D(x[i], y[i], z[i], x_[i], y_[i],0))
       
    
    R = np.mean(Rs[rotation_burn:min(rotation_burn+sample_size, len(Rs))], axis = 0)
    
    xr = []
    yr = []
    zr = []

    for i in range(0,len(x)):
        #X_t = rotateCoordSys3D(xrt[i],yrt[i],zrt[i],R)
        X_t = rotateCoordSys3D(x[i],y[i],z[i],R)
        xr.append(X_t[0][0])
        yr.append(X_t[1][0])
        zr.append(X_t[2][0])

x_s = x_[scale_burn:min(scale_burn + sample_size, len(x_))]
y_s = y_[scale_burn:min(scale_burn + sample_size, len(x_))]
z_s = z_[scale_burn:min(scale_burn + sample_size, len(x_))]

xr_s  = xr[scale_burn:min(scale_burn + sample_size, len(x_))]
yr_s = yr[scale_burn:min(scale_burn + sample_size, len(x_))]
zr_s = zr[scale_burn:min(scale_burn + sample_size, len(x_))]

scale_x = (np.max(np.abs(x_s)) - np.min(np.abs(x_s))) / (np.max(np.abs(xr_s)) - np.min(np.abs(xr_s)))
scale_y = (np.max(np.abs(y_s)) - np.min(np.abs(y_s))) / (np.max(np.abs(yr_s)) - np.min(np.abs(yr_s))) 
scale_z = (np.max(np.abs(z_s)) - np.min(np.abs(z_s))) / (np.max(np.abs(zr_s)) - np.min(np.abs(zr_s))) 

x_r_s = np.array(xr) * scale_x 
y_r_s = np.array(yr) * scale_y
z_r_s = np.array(zr) * scale_z

x_nr_s = np.array(x) * scale_x
y_nr_s = np.array(y) * scale_y
z_nr_s = np.array(z) * scale_z


err_x = np.abs(np.array(x_) - np.array(x_r_s))
err_y = np.abs(np.array(y_) - np.array(y_r_s))
err = []
for i in range(0,len(x)):
    err.append(np.sqrt((err_x[i])**2 + (err_y[i])**2))
            
if (plot_rot):
    plt.plot(xr, yr)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('NoGPS rotated')
    plt.show()
    
    plt.plot(x_, y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('GPS')
    plt.show()
    
if plot_rot_scale:
    plt.plot(x_r_s, y_r_s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('NoGPS rotated and scaled')
    plt.show()
    
    plt.plot(x_, y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('GPS')
    plt.show()
    
    plt.plot(x_r_s,y_r_s,x_,y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('GPS - NoGPS')
    plt.legend(['NoGPS', 'GPS'])
    plt.show()
    
    plt.plot(key_frames,x_r_s, key_frames,x_)
    plt.xlabel('KF')
    plt.ylabel('x')
    plt.title('NoGPS x - GPS x')
    plt.legend(['NoGPS', 'GPS'])
    plt.show()
    
    plt.plot(key_frames,y_r_s, key_frames,y_)
    plt.xlabel('KF')
    plt.ylabel('y')
    plt.title('NoGPS y - GPS y')
    plt.legend(['NoGPS', 'GPS'])
    plt.show()
    
    print('MSE x = ',MSE(x_r_s,x_))
    print('MSE y = ',MSE(y_r_s,y_))
    print('Max err x = ',np.max(err_x))
    print('Max err y = ',np.max(err_y))
    print('Max err = ',np.max(err))
    
    fig = plt.figure(4)
    #X, Y = np.meshgrid(data["x_offset"], data["y_offset"])
    ax = plt.axes(projection='3d')
    ax.plot3D(x_, y_, z_, 'gray')
    ax.scatter3D(x_r_s,y_r_s,z_r_s, cmap='Greens')
    
if plot_master:
    
    plt.plot(x,y,'--',x_,y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('Estimirana trajektorija letelice')
    plt.legend(['Estimacija', 'GPS'])
    plt.show()
    
    plt.plot(xr,yr,'--',x_,y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('Estimirana trajektorija letelice sa rotacijom')
    plt.legend(['Estimacija', 'GPS'])
    plt.show()
    
    plt.plot(x_r_s,y_r_s,'--',x_,y_)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('Estimirana trajektorija letelice sa rotacijom i skaliranjem')
    plt.legend(['Estimacija', 'GPS'])
    plt.show()
    
    plt.plot(key_frames,x_r_s,'--', key_frames,x_)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('x')
    plt.title('x koordinata')
    plt.legend(['Estimacija', 'GPS'])
    plt.show()
    
    plt.plot(key_frames,y_r_s,'--', key_frames,y_)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('y')
    plt.title('y koordinata')
    plt.legend(['Estimacija', 'GPS'])
    plt.show()
    
    print('MSE x = ',MSE(x_r_s,x_))
    print('MSE y = ',MSE(y_r_s,y_))
    print('Max err x = ',np.max(err_x))
    print('Max err y = ',np.max(err_y))
    print('Max err = ',np.max(err))
    print('Traj. path: ', traj_path)
    
    
if plot_data:
    
    plt.plot(cam_pitch2)
    plt.xlabel('n')
    plt.ylabel('Ugao [deg]')
    plt.title('Pitch kamere')
    plt.show()
    
    plt.plot(cam_yaw2)
    plt.xlabel('n')
    plt.ylabel('Ugao [deg]')
    plt.title('Yaw kamere')
    plt.show()
    
    plt.plot(drn_pitch2)
    plt.xlabel('n')
    plt.ylabel('Ugao [deg]')
    plt.title('Pitch letelice')
    plt.show()
    
    plt.plot(drn_roll2)
    plt.xlabel('n')
    plt.ylabel('Ugao [deg]')
    plt.title('Roll letelice')
    plt.show()
    
    plt.plot(drn_yaw2)
    plt.xlabel('n')
    plt.ylabel('Ugao [deg]')
    plt.title('Yaw letelice')
    plt.show()
    
    
    plt.plot(cam_pitch)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('Ugao [deg]')
    plt.title('Pitch kamere')
    plt.show()
    
    plt.plot(cam_yaw)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('Ugao [deg]')
    plt.title('Yaw kamere')
    plt.show()
    
    plt.plot(drn_pitch)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('Ugao [deg]')
    plt.title('Pitch letelice')
    plt.show()
    
    plt.plot(drn_roll)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('Ugao [deg]')
    plt.title('Roll letelice')
    plt.show()
    
    plt.plot(drn_yaw)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('Ugao [deg]')
    plt.title('Yaw letelice')
    plt.show()
    
    plt.plot(alt)
    plt.xlabel('Ključni frejmovi')
    plt.ylabel('Visina [m]')
    plt.title('Visina letelice')
    plt.show()
    
    

if plot_imu:
    plt.plot(imux, imuy)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('IMU')
    plt.show()
    
    plt.plot(xi, yi)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('GPS imu cutoff')
    plt.show()    
    
    plt.plot(axs)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('ax')
    plt.show()    

    plt.plot(ays)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('ay')
    plt.show()    
    
    plt.plot(vxs)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('vx')
    plt.show()    

    plt.plot(vys)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.title('vy')
    plt.show()    




