#!/bin/bash

# works on Raspbian 10 (buster)
# switch raspberry pi from adhoc mode to ethernet mode
# run this script as root (sudo -Es)


systemctl disable wpa_supplicant@wlan0.service
systemctl enable wpa_supplicant.service

# save as backup file
mv /etc/wpa_supplicant/wpa_supplicant-wlan0.conf /etc/wpa_supplicant/wpa_supplicant-wlan0.conf~

mv /etc/systemd/network/08-wifi.network /etc/systemd/network/08-wifi.network~

# disable systemd-networkd
systemctl disable systemd-networkd.service systemd-resolved.service
sed -i '1d' /etc/resolvconf.conf

# enable classic networking
mv /etc/network/interfaces~ /etc/network/interfaces
systemctl unmask networking.service dhcpcd.service

# recover /etc/resov.conf for dns service
mv /etc/resolv.conf.bak /etc/resolv.conf

# halt or reboot
reboot
