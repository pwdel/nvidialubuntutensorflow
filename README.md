# Installing Tensorflow on x64 Lubuntu

https://www.tensorflow.org/install/gpu

## Download the GeForce GTX Nvidia Driver

https://www.nvidia.com/download/driverResults.aspx/104284/en-us


## Disable toe Nouveau Kernal Driver

ERROR: The Nouveau kernel driver is currently in use by your system.  This driver is incompatible with the NVIDIA         
         driver, and must be disabled before proceeding.  Please consult the NVIDIA driver README and your Linux            
         distribution's documentation for details on how to correctly disable the Nouveau kernel driver.                    


For some distributions, Nouveau can be disabled by adding a file in the modprobe configuration directory.  Would you like nvidia-installer to attempt to create this modprobe file for you?


One or more modprobe configuration files to disable Nouveau have been written.  For some distributions, this may be sufficient to disable Nouveau; other distributions may require modification of the initial ramdisk.  Please reboot your system and attempt NVIDIA driver installation again.  Note if you later wish to reenable Nouveau, you will need to delete these files: /usr/lib/modprobe.d/nvidia-installer-disable-nouveau.conf, /etc/modprobe.d/nvidia-installer-disable-nouveau.conf      

  ERROR: Installation has failed.  Please see the file '/var/log/nvidia-installer.log' for details.  You may find           
         suggestions on fixing installation problems in the README available on the Linux driver download page at           
         www.nvidia.com.                                                                                                    
                           