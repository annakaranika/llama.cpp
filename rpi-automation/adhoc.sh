#!/bin/bash

# works on Raspbian 10 (buster)
# run this script as root (sudo -Es)
# switch raspberry pi from ethernet mode to adhoc mode
#   for first time setup, need to create 3 files before running the script:
#
#   (run as root)
#   /etc/systemd/network/08-wifi.network~:
#       cat > /etc/systemd/network/08-wifi.network~ <<EOF
#       [Match]
#       Name=wl*
#       [Network]
#       Address=192.168.4.1/24   # set the static ip address as you want, do not duplicate
#       EOF
#
#   /etc/wpa_supplicant/wpa_supplicant-wlan0.conf~:
#       cat > /etc/wpa_supplicant/wpa_supplicant-wlan0.conf <<EOF
#       ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
#       update_config=1
#       country=DE
#
#       network={
#               ssid="IBSS-RPiNet"
#               key_mgmt=NONE
#               mode=1
#               frequency=2412
#       }
#       EOF
#
#       chmod 600 /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
#       mv /etc/wpa_supplicant/wpa_supplicant-wlan0.conf /etc/wpa_supplicant/wpa_supplicant-wlan0.conf~
#
#
#   wpa_supplicant@wlan0.service:
#       systemctl edit wpa_supplicant@wlan0.service
#
#       # In the empty editor insert these statements, save it and quit the editor:
#
#       [Service]
#       ExecStart=
#       ExecStart=/sbin/wpa_supplicant -c/etc/wpa_supplicant/wpa_supplicant-%I.conf -Dwext -i%I
#


# disable classic networking
systemctl mask networking.service dhcpcd.service
mv /etc/network/interfaces /etc/network/interfaces~
sed -i '1i resolvconf=NO' /etc/resolvconf.conf

# enable systemd-networkd
systemctl enable systemd-networkd.service systemd-resolved.service
ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf

# use the pre-stored backup file
mv /etc/systemd/network/08-wifi.network~ /etc/systemd/network/08-wifi.network

# use new wpa config file
mv /etc/wpa_supplicant/wpa_supplicant-wlan0.conf~ /etc/wpa_supplicant/wpa_supplicant-wlan0.conf

systemctl disable wpa_supplicant.service
systemctl enable wpa_supplicant@wlan0.service

# halt or reboot
reboot
