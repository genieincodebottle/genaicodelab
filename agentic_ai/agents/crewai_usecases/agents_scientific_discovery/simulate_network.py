import random
import time

def generate_dummy_network_traffic():
    """Simulates network traffic logs with occasional threats."""
    threats = ["DDoS Attack", "Phishing Attempt", "Malware Injection", "SQL Injection", "Brute Force Login"]
    traffic_logs = []
    for i in range(10):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": f"192.168.1.{random.randint(1, 255)}",
            "destination_ip": "10.0.0.1",
            "status": "Normal Traffic"
        }
        # Introduce a threat in some entries
        if random.random() < 0.3:  # 30% chance of threat
            log_entry["status"] = random.choice(threats)
        traffic_logs.append(log_entry)
        time.sleep(0.5)
    return traffic_logs

def simulate_network():
    """Runs the network traffic simulation and prints logs."""
    print("Simulating network traffic...")
    logs = generate_dummy_network_traffic()
    for log in logs:
        print(log)
    return logs

if __name__ == "__main__":
    threat_logs = simulate_network()
    print("\nGenerated Threat Logs for AI Agents to Process:\n", threat_logs)
