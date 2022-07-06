#!/bin/bash

/usr/bin/nmcli -t -f SSID device wifi > /home/rms/wifi_list.conf
sort /home/rms/wifi_list.conf | uniq > /home/rms/ssid_list.conf
sed -i '/^$/d' /home/rms/ssid_list.conf
exit 0
