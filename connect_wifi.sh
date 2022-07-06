#!/bin/bash

# Usage: ./connect_wifi.sh "<SSID>" "<PASS>"

/usr/bin/nmcli dev wifi connect "$1" password "$2"
