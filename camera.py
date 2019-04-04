import picamera
from time import sleep
i = 0
n = 0
m = 0
camera = picamera.PiCamera()
while 1:
    i = input()
    if i == '1':
        n+=1
        camera.capture((str(n)+'image.jpg'))
        camera.start_preview()
        #camera.vflip = True
        camera.brightness = 60
##        sleep(3)
    if i == '2':
        m+=1
        camera.start_recording((str(m)+'video.h264'))
        camera.start_preview()
        #camera.vflip = True
        camera.brightness = 60
        sleep(20)
        print(finished)
        camera.stop_recording()
        camera.stop_preview()

