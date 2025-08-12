set -ex

# turn on IBGDA
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | tee -a /etc/modprobe.d/nvidia.conf

if command -v update-initramfs &> /dev/null; then
    # for Debian/Ubuntu
    sudo update-initramfs -u
elif command -v dracut &> /dev/null; then
    # for Fedora/CentOS
    sudo dracut --force
else
    echo "No supported initramfs update tool found."
    exit 1
fi

echo "Please reboot the system to apply the changes"
