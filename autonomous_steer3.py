import cv2
import torch
import numpy as np
from ultralytics import YOLO
import logging
import time
import can
import cantools
import asyncio
import copy
import crc8
from typing import Tuple, Optional
from gearshifter import GearController, GearPositions


class MCMController:
    """Motion Control Module interface for vehicle control"""

    def __init__(self, channel):
        self.logger = logging.getLogger(__name__)
        self.db = cantools.database.Database()

        try:
            self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
            self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
            self.db.add_dbc_file('./sygnal_dbc/mcm/Override.dbc')
            self.logger.info("Loaded DBC files successfully")
        except Exception as e:
            self.logger.error(f"Failed to load DBC files: {e}")
            raise

        try:
            self.bus = can.Bus(bustype='socketcan',
                               channel=channel,
                               bitrate=500000)
            self.logger.info(f"CAN bus initialized on channel {channel}")
        except Exception as e:
            self.logger.error(f"Failed to initialize CAN bus: {e}")
            raise

        self.control_count = 0
        self.bus_address = 1
        self.control_states = {
            'steer': False,
            'throttle': False,
            'brake': False
        }

    def calc_crc8(self, data):
        """Calculate CRC8 for CAN message"""
        data_copy = copy.copy(data)
        del data_copy[7]
        hash = crc8.crc8()
        hash.update(data_copy)
        return hash.digest()[0]

    async def control_request(self, module, request):
        """Send control request message for any module"""
        try:
            control_enable_msg = self.db.get_message_by_name('ControlEnable')
            enable = 1 if request else 0
            interface = {'brake': 0, 'throttle': 1, 'steer': 2}[module]

            data = bytearray(control_enable_msg.encode({
                'BusAddress': self.bus_address,
                'InterfaceID': interface,
                'Enable': enable,
                'CRC': 0
            }))
            data[7] = self.calc_crc8(data)

            msg = can.Message(
                arbitration_id=control_enable_msg.frame_id,
                is_extended_id=False,
                data=data
            )
            self.bus.send(msg)
            self.logger.info(f"Sent control request: {msg}")

            await asyncio.sleep(0.05)
            self.control_states[module] = request
            self.logger.debug(f"Control request sent: {module} {'enabled' if request else 'disabled'}")

        except Exception as e:
            self.logger.error(f"Control request error: {e}")
            raise

    async def update_setpoint(self, module, value):
        """Send control command message for any module"""
        try:
            if self.control_states[module]:
                control_cmd_msg = self.db.get_message_by_name('ControlCommand')
                interface = {'brake': 0, 'throttle': 1, 'steer': 2}[module]

                # Ensure value is within bounds [-1, 1]
                value = max(min(value, 1.0), -1.0)

                data = bytearray(control_cmd_msg.encode({
                    'BusAddress': self.bus_address,
                    'InterfaceID': interface,
                    'Count8': self.control_count,
                    'Value': value,
                    'CRC': 0
                }))
                data[7] = self.calc_crc8(data)

                msg = can.Message(
                    arbitration_id=control_cmd_msg.frame_id,
                    is_extended_id=False,
                    data=data
                )
                self.bus.send(msg)
                self.logger.info(f"Sent {module} command: {msg}")

                self.control_count = (self.control_count + 1) % 256
                await asyncio.sleep(0.05)
                self.logger.debug(f"{module.capitalize()} setpoint updated: {value}")

        except Exception as e:
            self.logger.error(f"Setpoint update error: {e}")
            raise

    async def send_heartbeat_clear_seed(self, subsystem_id):
        """Send HeartbeatClearSeed message for specified subsystem"""
        try:
            heartbeat_clear_msg = self.db.get_message_by_name('HeartbeatClearSeed')
            seed = int(time.time() * 1000) % 0xFFFFFFFF  # Use timestamp as seed

            data = bytearray(heartbeat_clear_msg.encode({
                'BusAddress': self.bus_address,
                'SubsystemID': subsystem_id,
                'ResetSeed': seed,
                'CRC': 0
            }))
            data[7] = self.calc_crc8(data)

            msg = can.Message(
                arbitration_id=heartbeat_clear_msg.frame_id,
                is_extended_id=False,
                data=data
            )
            self.bus.send(msg)
            self.logger.info(f"Sent HeartbeatClearSeed for subsystem {subsystem_id}")
            await asyncio.sleep(0.05)  # Small delay between messages

        except Exception as e:
            self.logger.error(f"Failed to send HeartbeatClearSeed: {e}")
            raise

    async def send_heartbeat_clear_key(self, subsystem_id):
        """Send HeartbeatClearKey message for specified subsystem"""
        try:
            heartbeat_clear_msg = self.db.get_message_by_name('HeartbeatClearKey')
            key = int(time.time() * 1000) % 0xFFFFFFFF  # Use timestamp as key

            data = bytearray(heartbeat_clear_msg.encode({
                'BusAddress': self.bus_address,
                'SubsystemID': subsystem_id,
                'ResetKey': key,
                'CRC': 0
            }))
            data[7] = self.calc_crc8(data)

            msg = can.Message(
                arbitration_id=heartbeat_clear_msg.frame_id,
                is_extended_id=False,
                data=data
            )
            self.bus.send(msg)
            self.logger.info(f"Sent HeartbeatClearKey for subsystem {subsystem_id}")
            await asyncio.sleep(0.05)  # Small delay between messages

        except Exception as e:
            self.logger.error(f"Failed to send HeartbeatClearKey: {e}")
            raise

    async def clear_subsystem_state(self, subsystem_id):
        """Clear state for a specific subsystem using seed-key sequence"""
        try:
            # Send seed first
            await self.send_heartbeat_clear_seed(subsystem_id)
            await asyncio.sleep(0.1)  # Wait for seed to be processed

            # Then send key
            await self.send_heartbeat_clear_key(subsystem_id)
            await asyncio.sleep(0.1)  # Wait for key to be processed

            self.logger.info(f"Completed state clearing for subsystem {subsystem_id}")

        except Exception as e:
            self.logger.error(f"Failed to clear state for subsystem {subsystem_id}: {e}")
            raise
class AutoSteeringSystem:
    """Autonomous steering control system with integrated gear management"""

    def __init__(self, model_path: str, can_channel: str = 'can0'):
        # Initialize logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        try:
            self.gear_controller = GearController(can_channel)
            self.model = YOLO(model_path)
            self.model.to('cuda')
            self.logger.info(f"Using device: {self.model.device}")
            self.mcm = MCMController(can_channel)
            self.logger.info("All controllers initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise

        # Lane detection parameters
        self.ym_per_pix = 17 / 720
        self.xm_per_pix = 17 / 1280
        self.lane_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]

        # Steering control parameters
        self.max_torque = 0.5
        self.min_curvature_radius = 11
        self.torque_smoothing = 0.7
        self.prev_torque = 0.0

        # Throttle control parameters
        self.max_throttle = 0.000
        self.min_throttle = 0.000
        self.throttle_smoothing = 0.00
        self.prev_throttle = 0.0

        # Brake control parameters
        self.permanent_brake_force = 0.1
        self.brake_threshold_radius = 11
        self.max_brake_force = 0.5
        self.min_brake_duration = 0.5
        self.brake_release_duration = 0.3
        self.is_braking = True
        self.brake_start_time = time.time()
        self.last_brake_time = 0
        self.brake_cooldown = 2.0

        self.tight_turn_threshold = 9.0  # Radius threshold for temporary brake release
        self.temp_brake_release_duration = 2.0  # Duration in seconds to release brakes
        self.temp_release_start_time = 0.0
        self.is_temp_release_active = False
        self.min_time_between_releases = 4.0  # Minimum time between temporary releases
        self.last_temp_release_time = 0.0

    def calculate_torque(self, radius: float) -> float:
        """Calculate steering torque based on curve radius"""
        if radius == float('inf'):
            return 0.0

        direction = 1 if radius > 0 else -1
        radius = abs(radius)

        if radius < self.min_curvature_radius:
            torque = self.max_torque
        else:
            torque = self.max_torque * np.exp(-(radius - self.min_curvature_radius) / 90.0)

        # Apply smoothing
        torque = (self.torque_smoothing * self.prev_torque +
                  (1 - self.torque_smoothing) * torque)
        self.prev_torque = torque

        return direction * min(abs(torque), self.max_torque)

    def calculate_curvature(self, lane_points: np.ndarray) -> float:
        """Calculate the radius of curvature in meters with direction"""
        if len(lane_points) < 3:
            return float('inf')

        try:
            points = np.array(lane_points)
            y_points = points[:, 1]
            x_points = points[:, 0]

            # Sort and remove duplicates
            sort_idx = np.argsort(y_points)
            y_points = y_points[sort_idx]
            x_points = x_points[sort_idx]
            _, unique_idx = np.unique(y_points, return_index=True)
            y_points = y_points[unique_idx]
            x_points = x_points[unique_idx]

            if len(y_points) < 3:
                return float('inf')

            # Convert to real-world coordinates
            y_eval = np.max(y_points) * self.ym_per_pix
            coeffs = np.polyfit(y_points * self.ym_per_pix,
                                x_points * self.xm_per_pix,
                                2)

            # Handle nearly straight lines
            if abs(coeffs[0]) < 1e-4:
                return float('inf')

            # Calculate basic curvature
            curvature = ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) * \
                        np.absolute(2 * coeffs[0]) ** -1

            # Determine direction from coefficients
            direction = 1 if coeffs[0] > 0 else -1
            curvature = curvature * direction

            return min(abs(curvature), 1000.0) * direction

        except Exception as e:
            self.logger.warning(f"Curvature calculation error: {str(e)}")
            return float('inf')

    async def calculate_throttle(self, curve_radius: float) -> float:
        """Calculate throttle based on curve radius"""
        if abs(curve_radius) < self.brake_threshold_radius:
            throttle = self.min_throttle
        else:
            throttle = min(
                self.max_throttle,
                self.min_throttle + (self.max_throttle - self.min_throttle) *
                (abs(curve_radius) - self.brake_threshold_radius) / 20.0
            )

        # Apply smoothing
        throttle = self.throttle_smoothing * self.prev_throttle + \
                   (1 - self.throttle_smoothing) * throttle
        self.prev_throttle = throttle

        return throttle

    async def release_brakes(self):
        """Modified to maintain permanent brake force instead of releasing"""
        try:
            await self.mcm.update_setpoint('brake', self.permanent_brake_force)
        except Exception as e:
            self.logger.error(f"Brake adjustment failed: {e}")


    async def initialize_drive_mode(self):
        """Initialize the vehicle in drive mode with proper state verification"""
        try:
            self.logger.info("Initializing drive mode")

            # Clear states for both subsystems before gear change
            # This ensures clean state initialization before any mechanical operations
            for subsystem_id in [0, 1]:
                self.logger.info(f"Clearing initial state for subsystem {subsystem_id}")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.2)  # Ensure complete processing

            # Apply initial brake for safety during gear change
            await self.mcm.control_request('brake', True)
            await self.mcm.update_setpoint('brake', self.permanent_brake_force)
            await asyncio.sleep(0.2)

            # Execute gear change to Drive
            self.logger.info("Shifting to Drive mode")
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.DRIVE,
                brake_percentage=0.8,
                brake_duration=1.0
            )

            await self.mcm.update_setpoint('brake', self.permanent_brake_force)
            self.is_braking = True

            # Verify system state after gear change
            # This ensures clean state after mechanical transition
            for subsystem_id in [0, 1]:
                self.logger.info(f"Verifying post-gear state for subsystem {subsystem_id}")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.1)

            self.logger.info("Drive mode initialization completed")
            await asyncio.sleep(0.5)  # Allow time for final gear engagement

        except Exception as e:
            self.logger.error(f"Drive mode initialization failed: {e}")
            raise

    async def safe_stop_sequence(self):
        """Execute safe stop sequence with proper state clearing"""
        try:
            self.logger.info("Executing safe stop sequence")

            # First shift to Neutral with moderate braking
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.NEUTRAL,
                brake_percentage=0.4,
                brake_duration=1.5
            )
            await asyncio.sleep(0.5)  # Allow for gear disengagement

            # Then shift to Park with firm braking
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.PARK,
                brake_percentage=0.5,
                brake_duration=2.0
            )

            # Clear states for both subsystems with proper seed-key sequence
            for subsystem_id in [0, 1]:
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.2)  # Ensure complete processing between subsystems

            self.logger.info("Safe stop sequence completed")
        except Exception as e:
            self.logger.error(f"Error during safe stop sequence: {e}")
            raise


    async def coordinate_controls(self, curve_radius: float) -> Tuple[float, float, float]:
        """Coordinate steering, throttle, and brake controls with temporary brake release for tight turns"""
        current_time = time.time()

        # Check for tight turn condition
        if abs(curve_radius) < self.tight_turn_threshold and \
                current_time - self.last_temp_release_time > self.min_time_between_releases and \
            not self.is_temp_release_active:
            # Initiate temporary brake release
                self.is_temp_release_active = True
                self.temp_release_start_time = current_time
                self.last_temp_release_time = current_time
                self.logger.info(f"Initiating temporary brake release for tight turn (radius: {curve_radius:.1f}m)")

                # Calculate strong steering input for tight turn
                steering = self.calculate_torque(curve_radius)
                return (steering, 0.0, 0.0)  # Release brakes completely during initial turn

        # Check if we're in an active temporary release
        if self.is_temp_release_active:
            release_duration = current_time - self.temp_release_start_time

            if release_duration >= self.temp_brake_release_duration:
                self.is_temp_release_active = False
                self.logger.info("Ending temporary brake release, returning to permanent brake force")
                return (0.0, 0.0, self.permanent_brake_force)
            else:
                # During temporary release, allow steering but no throttle
                steering = self.calculate_torque(curve_radius)
                return (steering, 0.0, 0.0)

        # Normal operation
        steering = self.calculate_torque(curve_radius)
        throttle = await self.calculate_throttle(curve_radius)
        return (steering, throttle, self.permanent_brake_force)

    async def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
        """Process frame and return control signals"""
        try:
            resized_frame = cv2.resize(frame, (640, 640))
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
            frame_tensor /= 255.0

            results = self.model(frame_tensor, verbose=False)
            annotated_frame = frame.copy()
            radius = float('inf')

            if results[0].masks:
                masks = results[0].masks.data
                if len(masks) > 0:
                    mask = masks[-1]
                    mask_array = mask.cpu().numpy()
                    mask_uint8 = (mask_array * 255).astype(np.uint8)

                    if mask_uint8.shape[:2] != frame.shape[:2]:
                        mask_uint8 = cv2.resize(mask_uint8, (frame.shape[1], frame.shape[0]))

                    kernel = np.ones((3, 3), np.uint8)
                    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_TC89_KCOS)

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        num_points = 10
                        indices = np.linspace(0, len(largest_contour) - 1, num_points).astype(int)
                        curve_points = np.array([largest_contour[i][0] for i in indices])
                        radius = self.calculate_curvature(curve_points)
                        cv2.polylines(annotated_frame, [largest_contour], False,
                                         self.lane_colors[0], 2)

            # Get coordinated control signals
            steering, throttle, brake = await self.coordinate_controls(radius)

            # Draw information overlay
            height = frame.shape[0]
            info_text = [
                f'Curve Radius: {radius:.1f}m',
                f'Steering: {steering:.3f}',
                f'Throttle: {throttle:.3f}',
                f'Brake: {brake:.3f}',
                f'Braking: {"Yes" if self.is_braking else "No"}',
                f'Temp Release: {"Active" if self.is_temp_release_active else "Inactive"}'
            ]

            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text,
                           (20, height - 30 * (len(info_text) - i)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            return annotated_frame, (steering, throttle, brake)

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame, None

    async def run(self, source=0):
        """Main control loop with optimized video capture and proper brake management"""
        self.logger.info(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return

        # Configure video capture for optimal performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffering latency

        # Verify camera configuration
        actual_format = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        self.logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        if actual_format != cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'):
            self.logger.warning("Failed to set MJPG format - performance may be suboptimal")

        try:
            # Initialize drive mode
            await self.initialize_drive_mode()

            # Enable all controls while maintaining brake state
            for control in ['steer', 'throttle', 'brake']:
                await self.mcm.control_request(control, True)
                self.logger.info(f"{control.capitalize()} control enabled")

            # Ensure permanent brake force is applied initially
            await self.mcm.update_setpoint('brake', self.permanent_brake_force)

            self.logger.info("Starting main control loop")
            frame_count = 0
            start_time = time.time()

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.logger.error("Failed to read frame")
                    break

                frame_count += 1
                if frame_count % 30 == 0:  # Log FPS every 30 frames
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    self.logger.info(f"Current FPS: {current_fps:.2f}")

                annotated_frame, controls = await self.process_frame(frame)

                if controls:
                    steering, throttle, brake = controls
                    try:
                        # Apply controls while respecting brake states
                        await self.mcm.update_setpoint('brake', brake)
                        await self.mcm.update_setpoint('steer', steering)
                        await asyncio.sleep(0.02)  # Brief delay between control updates
                        await self.mcm.update_setpoint('throttle', throttle)
                    except Exception as e:
                        self.logger.error(f"Control update failed: {e}")
                        # Ensure brakes are applied on error
                        await self.mcm.update_setpoint('brake', self.permanent_brake_force)

                # Update display with control information
                if annotated_frame is not None:
                    fps_text = f"FPS: {frame_count / (time.time() - start_time):.1f}"
                    cv2.putText(annotated_frame, fps_text, (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Autonomous Driving Window', annotated_frame)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    self.logger.info("User requested exit")
                    break

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            # Always ensure brakes are applied during shutdown
            self.logger.info("Initiating shutdown sequence")
            try:
                # Sequence control shutdown
                await self.mcm.update_setpoint('throttle', 0)
                await asyncio.sleep(0.1)
                await self.mcm.update_setpoint('brake', self.permanent_brake_force)
                await asyncio.sleep(0.1)
                await self.mcm.update_setpoint('steer', 0)
                await asyncio.sleep(0.1)

                await self.safe_stop_sequence()

                # Disable all controls in reverse order
                for control in ['throttle', 'brake', 'steer']:
                    await self.mcm.control_request(control, False)
                    await asyncio.sleep(0.05)

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Shutdown complete")


def main():
    """Main entry point with argument parsing and system initialization"""
    logging.info("Starting Autonomous Steering System")

    import argparse
    parser = argparse.ArgumentParser(description='Autonomous Steering Control System')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model weights')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--can-channel', type=str, default='can0',
                       help='CAN bus channel for vehicle control')
    args = parser.parse_args()

    try:
        system = AutoSteeringSystem(args.model, args.can_channel)
        asyncio.run(system.run(0 if args.source == '0' else args.source))
    except Exception as e:
        logging.error(f"System failed to start: {e}")
        raise


if __name__ == '__main__':
    main()