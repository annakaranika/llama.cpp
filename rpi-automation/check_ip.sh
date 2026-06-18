iw dev
iw dev wlan0 link
ip addr
sudo iw dev wlan0 scan | grep -B8 -A3 "IBSS-RPiNet"