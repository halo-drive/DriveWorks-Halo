import asyncio
import can
import cantools
import crc8
import logging
from enum import Enum
from pkg_resources import resource_filename
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class GearPosition:
    """Gear position configuration with specific voltage/position values"""
    name: str
    value: float
    interface: int = 0  # Default interface for gear control


class GearPositions:
    """Available gear positions with their corresponding control values"""
    PARK = GearPosition("PARK", 0.0)
    REVERSE = GearPosition("REVERSE", 56.0)
    DRIVE = GearPosition("DRIVE", 40.0)
    NEUTRAL = GearPosition("NEUTRAL", 48.0)

    @classmethod
    def get_all_positions(cls):
        return [cls.PARK, cls.REVERSE, cls.DRIVE, cls.NEUTRAL]

    @classmethod
    def from_name(cls, name: str) -> GearPosition:
        """Get gear position by name"""
        positions = {
            'P': cls.PARK,
            'R': cls.REVERSE,
            'D': cls.DRIVE,
            'N': cls.NEUTRAL
        }
        return positions.get(name.upper())


class MCM:
    """Motion Control Module for brake control"""

    def __init__(self, channel):
        self.db = cantools.database.Database()
        self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        self.bus = can.Bus(bustype='socketcan', channel=channel, bitrate=500000)
        self.control_count = 0
        self.bus_address = 1

    def calc_crc8(self, data):
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]

    async def enable_brake_control(self):
        control_enable_msg = self.db.get_message_by_name('ControlEnable')
        data = bytearray(control_enable_msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 0,  # Brake interface
            'Enable': 1,
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_enable_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        logging.info("Brake control enabled")
        await asyncio.sleep(0.1)

    async def update_brake_setpoint(self, value):
        control_cmd_msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(control_cmd_msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 0,
            'Count8': self.control_count,
            'Value': value,
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_cmd_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        logging.info(f"Brake setpoint updated to {value}")
        self.control_count = (self.control_count + 1) % 256


class CB:
    """Control Box for gear control"""

    def __init__(self, channel):
        self.db = cantools.database.Database()
        self.db.add_dbc_file(resource_filename('sygnal_dbc.cb', 'Heartbeat.dbc'))
        self.db.add_dbc_file(resource_filename('sygnal_dbc.cb', 'Control.dbc'))
        self.bus = can.Bus(bustype='socketcan', channel=channel, bitrate=500000)
        self.control_count = 0
        self.bus_address = 3  # CB bus address for gear control

    def calc_crc8(self, data):
        data_copy = bytearray(data[:-1])
        hash = crc8.crc8()
        hash.update(data_copy)
        return hash.digest()[0]

    async def enable_control(self, interface):
        control_enable_msg = self.db.get_message_by_name('ControlEnable')
        data = bytearray(control_enable_msg.encode({
            'BusAddress': self.bus_address,
            'MessageID': interface,
            'Enable': 1,
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_enable_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        logging.info(f"Gear control enabled for interface {interface}")
        await asyncio.sleep(0.1)

    async def update_gear_setpoint(self, gear_position: GearPosition):
        control_cmd_msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(control_cmd_msg.encode({
            'BusAddress': self.bus_address,
            'MessageID': gear_position.interface,
            'Count8': self.control_count,
            'Value': gear_position.value,
            'CRC': 0
        }))
        data[-1] = self.calc_crc8(data)
        msg = can.Message(arbitration_id=control_cmd_msg.frame_id, data=data, is_extended_id=False)
        self.bus.send(msg)
        logging.info(f"Gear setpoint updated to {gear_position.name} ({gear_position.value})")
        self.control_count = (self.control_count + 1) % 256


class GearController:
    """Coordinated control of brakes and gear shifting"""

    def __init__(self, channel='can0'):
        self.mcm = MCM(channel)
        self.cb = CB(channel)

    async def execute_gear_change(self, gear_position: GearPosition, brake_percentage=0.6, brake_duration=1.0,
                                  gear_command_interval=0.1):
        """
        Execute a gear change with the following sequence:
        1. Apply brakes
        2. Send gear change command repeatedly while brakes are applied
        3. Release brakes after duration
        """
        logging.info(f"Starting gear change sequence to {gear_position.name}")

        # Enable controls
        await self.mcm.enable_brake_control()
        await self.cb.enable_control(gear_position.interface)

        # Apply brakes
        await self.mcm.update_brake_setpoint(brake_percentage)
        logging.info(f"Brakes applied at {brake_percentage * 100}%")

        # Calculate number of gear commands to send
        num_commands = int(brake_duration / gear_command_interval)

        # Send gear commands while brakes are applied
        for _ in range(num_commands):
            await self.cb.update_gear_setpoint(gear_position)
            await asyncio.sleep(gear_command_interval)

        # Release brakes
        await self.mcm.update_brake_setpoint(0.0)
        logging.info("Brakes released")


async def main():
    controller = GearController()

    print("\nAvailable gear positions:")
    for pos in GearPositions.get_all_positions():
        print(f"{pos.name}: {pos.value}")

    while True:
        gear_input = input("\nEnter gear position (P/R/N/D or Q to quit): ").upper()
        if gear_input == 'Q':
            break

        gear_position = GearPositions.from_name(gear_input)
        if not gear_position:
            print("Invalid gear position. Please use P, R, N, or D.")
            continue

        brake_force = float(input("Enter brake force percentage (0-100): ")) / 100.0
        brake_duration = float(input("Enter brake application duration (seconds): "))

        print(f"\nInitiating gear change sequence to {gear_position.name}...")
        await controller.execute_gear_change(
            gear_position=gear_position,
            brake_percentage=brake_force,
            brake_duration=brake_duration
        )


if __name__ == "__main__":
    asyncio.run(main())