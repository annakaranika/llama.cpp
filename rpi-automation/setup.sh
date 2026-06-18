if [ "${#@}" -eq 0 ]; then
  echo "Please provide id of pi. E.g.: ./setup.sh 1"
  exit
fi

# enable and start ssh
systemctl enable ssh
systemctl start ssh

# disable classic networking
systemctl mask networking.service dhcpcd.service
mv /etc/network/interfaces /etc/network/interfaces~
sed -i '1i resolvconf=NO' /etc/resolvconf.conf

# enable systemd-networkd
systemctl enable systemd-networkd.service systemd-resolved.service
ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf

# set its own ip
cat > /etc/systemd/network/08-wifi.network <<EOF
[Match]
Name=wl*
[Network]
Address=192.168.4.$1/24
EOF

# Setup unprotected ad-hoc interface using wpa_supplicant

cat > /etc/wpa_supplicant/wpa_supplicant-wlan0.conf <<EOF
country=DE
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="IBSS-RPiNet"
    frequency=2412
    mode=1   # IBSS (ad-hoc, peer-to-peer)
    key_mgmt=NONE
}
EOF

chmod 600 /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
systemctl disable wpa_supplicant.service
systemctl enable wpa_supplicant@wlan0.service

SYSTEMD_EDITOR=tee systemctl edit wpa_supplicant@wlan0.service <<EOF
[Service]
ExecStart=
ExecStart=/sbin/wpa_supplicant -c/etc/wpa_supplicant/wpa_supplicant-%I.conf -Dwext -i%I
EOF

reboot