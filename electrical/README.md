## Motion Control Board

The Motion Control Board (MCB) is a carrier board designed for the Radxa CM5, providing comprehensive connectivity for robotic applications.

- The board serves as a carrier board for the Radxa CM5.
- Power is supplied via a 5V input through an XT30 connector.
- A UART TTL debug port is available for serial debugging purposes.
- The LSM6DSV IMU is located at the bottom of the board, with an I2C connector available if an external IMU is required.
- All signal connections use JST-GH connectors
- LCSC component IDs are attached in the KiCad source files for direct JLCPCB fabrication and assembly.

### Communication Ports

| Port Type | Quantity | Details |
|-----------|----------|---------|
| CAN (Native) | 3 | Direct CAN interfaces |
| CAN (SPI-CAN) | 3 | CAN via SPI bridge |
| USB 2.0 | 3 | USB interfaces |
| Ethernet | 1 | 1Gbps |
| RS485 | 2 | Serial communication |
| I2C | 1 | I2C interface |

### Port Mapping

| Port ID | Purpose |
|---------|---------|
| C0 | Left leg bus |
| C1 | Right leg bus |
| C2 | Right arm bus |
| C3 | Left arm bus |
| C4 | Waist-neck bus |
| C5 | BMS bus |

> The waist-neck bus requires a different cabling setup for this revision of the MCB: Waist -> Neck yaw -> Neck pitch -> MCB port C4

## Wiring

![WireViz](https://github.com/wireviz/WireViz) was utilised for representing the wiring harness of the robot. 

Please refer to WireViz documentation for available visualisation options.
### Cable Naming Convention

```
W-<ID>-<TYPE>
ID = unique ID
TYPE = Power (PWR) or Signal (SIG) cable bundle
```



