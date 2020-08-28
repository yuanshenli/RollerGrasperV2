# Roller Grasper V2
This repository contains implementations for the paper titled "Design and Control of Roller Grasper V2 for In-Hand Manipulation". Further information can be found at [the project page](https://yuanshenli.com/roller_grasper_v2.html)

## Simulation 
Uses [MuJoCo](http://www.mujoco.org/) physics engine and [mujoco-py](https://github.com/openai/mujoco-py) python wrapper.

Handcrafted policy demos:
```
python handcrafted_demos.py
```

Imitation learning:
```
python imitation_learning.py
```

## Real-world experiment
Firmware and serial communication API can be found at [rgSerial](https://github.com/yuanshenli/rgSerial).

## Mechanical Design
Coming soon... 

### Electronics
The circuit diagram can be found in the paper. 

[Teensy 3.6](http://www.robotis.us/dynamixel-xh430-w350-t/): handles serial communication and low level controllers

[SMPS2Dynamixel Adapter](https://www.trossenrobotics.com/store/p/5886-SMPS2Dynamixel-Adapter.aspx): used as power regulator for the whole circuit

[Dynamixel DYNAMIXEL XM430-W350-T](http://www.robotis.us/dynamixel-xh430-w350-t/): base joints

[Micro gear motors](https://www.servocity.com/110-rpm-micro-gear-motor-w-encoder): pivot joints and roller joints

[Dual MC33926 Motor Driver Carrier](https://www.pololu.com/product/1213): dirve the Micro gear motors

## 
If you find this work helpful, please cite with the following bibtex:

```
@INPROCEEDINGS{rollergrasperV2,  
author={S. {Yuan} and L. {Shao} and C. L. {Yako} and A. {Gruebele} and J. K. {Salisbury}},  
booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},   
title={Design and Control of Roller Grasper V2 for In-Hand Manipulation},   
year={2020}}
```
