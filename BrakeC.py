import asyncio
import can
import cantools
import crc8
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class MCM:
    def __init__(self, channel):
        self.db = cantools.database.Database()
        self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        self.bus = can.Bus(bustype='socketcan', channel=channel, bitrate=500000)
        self.control_count = 0
        self.bus_address = 1  # Assuming MCM bus address

    def calc_crc8(self, data):
        hash = crc8.crc8()
        hash.update(data[:-1])  # Exclude CRC byte itself
        return hash.digest()[0]

    async def enable_control(self, module):
        control_enable_msg = self.db.get_message_by_name('ControlEnable')
        interface = {'brake': 0, 'accel': 1, 'steer': 2}.get(module)
        data = bytearray(control_enable_msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': interface,
            'Enable': 1,  # Enable the control
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_enable_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        await asyncio.sleep(0.1)  # Wait for confirmation

    async def update_brake_setpoint(self, value):
        control_cmd_msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(control_cmd_msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 0,  # Brake interface
            'Count8': self.control_count,
            'Value': value,  # Apply brake force
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_cmd_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        logging.info(f"Sent brake command with ID: {msg.arbitration_id}, data: {msg.data.hex()}")
        self.control_count = (self.control_count + 1) % 256  # Increment or reset count

    async def apply_brakes(self, percentage, duration):
        await self.enable_control('brake')
        await self.update_brake_setpoint(percentage)  # Apply specified brake force
        await asyncio.sleep(duration)
        await self.update_brake_setpoint(0)  # Release brakes

# Run the braking code
async def main():
    mcm = MCM(channel='can0')
    
    # Request user input
    percentage = float(input("Enter brake percentage to be applied (e.g., 0.5 for 50%): "))
    duration = float(input("Enter the duration for which to apply the brakes in seconds: "))

    print(f"Applying brakes at {percentage * 100}% for {duration} seconds")
    await mcm.apply_brakes(percentage, duration)


asyncio.run(main())
