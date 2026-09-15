import os
import time
import json
import random

# File Settings for Inter-Process Communication (IPC)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SHARED_DATA_FILE = os.path.join(DATA_DIR, 'shared_data.json')

def simulate_sensor_data():
    """
    Simulates motor parameters from the AI4I 2020 dataset.
    Columns: Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear.
    Creates failure scenarios by gradually adjusting values to simulate wear/stress.
    Writes telemetry values into a JSON file for the Streamlit Dashboard.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Initial (Normal) Values on AI4I scale (e.g., Temp in Kelvin)
    machine_type = random.choice(['L', 'M', 'H'])
    air_temp = 298.0  # ~25 C
    process_temp = 308.0 # ~35 C
    rotational_speed = 1500 # RPM
    torque = 40.0 # Nm
    tool_wear = 0 # Minutes
    
    print(f"Simulator Publisher started. Saving to: {SHARED_DATA_FILE}")
    print("Machines are operating in normal (healthy) condition...")
    print("-" * 50)
    
    fault_mode_active = False
    cycle_count = 0
    
    try:
        while True:
            # Natural sensor noise/vibrations
            air_temp += random.uniform(-0.1, 0.1)
            process_temp += random.uniform(-0.2, 0.2)
            rotational_speed += random.randint(-15, 15)
            torque += random.uniform(-0.5, 0.5)
            tool_wear += random.randint(1, 3) # Wear increases slightly every cycle
            
            # Anomaly Scenario Trigger (trigger failure after 25 cycles)
            cycle_count += 1
            if cycle_count > 25 and not fault_mode_active:
                print("\n[WARNING] Abnormal wear/heat scenario activated. Machine issue detected!\n")
                fault_mode_active = True
                
            if fault_mode_active:
                # Motor under stress: Temp increases, speed drops, torque spikes
                process_temp += random.uniform(0.5, 1.5)
                rotational_speed -= random.randint(20, 50)
                torque += random.uniform(2.0, 5.0)
                tool_wear += random.randint(5, 15)
            
            # Reset values to realistic limits to prevent negative infinity and repetitive loops
            if rotational_speed < 0:
                rotational_speed = 0
            if process_temp > 400:
                process_temp = 400.0
            if torque > 100:
                torque = 100.0
                
            # Restart normal simulation after some time
            if cycle_count > 60:
                print("\n[INFO] Maintenance performed. Machine operating normally again.\n")
                fault_mode_active = False
                cycle_count = 0
                air_temp = 298.0
                process_temp = 308.0
                rotational_speed = 1500
                torque = 40.0
                tool_wear = 0
            
            # Construct Payload ('Type' is categorical)
            payload = {
                "Timestamp_raw": time.time(), # Hidden timestamp for distincting updates
                "Type": machine_type,
                "Air temperature": round(air_temp, 1),
                "Process temperature": round(process_temp, 1),
                "Rotational speed": int(rotational_speed),
                "Torque": round(torque, 1),
                "Tool wear": int(tool_wear)
            }
            
            # Write to JSON File
            with open(SHARED_DATA_FILE, 'w') as f:
                json.dump(payload, f)
            
            # Log output to console
            status_text = "NORMAL" if not fault_mode_active else "RISK OF FAILURE"
            print(f"[{status_text}] Data Sent: {payload['Torque']} Nm, {payload['Rotational speed']} RPM")
            
            time.sleep(1) # Publish 1 packet per second
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    simulate_sensor_data()
