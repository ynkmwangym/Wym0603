# -*- coding: utf-8  -*-

import RPi.GPIO as GPIO
import time

ledpin = 15
GPIO.setmode(GPIO.BOARD)  #set pin 15=wiringPi pin 3。
GPIO.setup(ledpin, GPIO.OUT)  #set pinmode as OUT
GPIO.output(ledpin, GPIO.HIGH)
GPIO.output(ledpin, GPIO.LOW) #test the LED

while True:
    GPIO.output(ledpin, True)  
    time.sleep(1)             #on for 1s
    GPIO.output(ledpin, False)  
    time.sleep(1)             #off for 1s
