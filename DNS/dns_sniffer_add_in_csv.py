from scapy.all import sniff, DNS, DNSQR
from datetime import datetime
import csv

# Create/open a CSV file and write headers
csv_file = open("dns_logs.csv", "a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "src_ip", "dst_ip", "query_domain", "query_type"])

def process_packet(packet):
    if packet.haslayer(DNSQR):
        log = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": packet[1].src,
            "dst_ip": packet[1].dst,
            "query_domain": packet[DNSQR].qname.decode(errors="ignore"),
            "query_type": packet[DNSQR].qtype
        }
        print(f"{log['timestamp']},{log['src_ip']},{log['dst_ip']},{log['query_domain']},{log['query_type']}")
        csv_writer.writerow(log.values())
        csv_file.flush()  # Save to disk continuously

print("Sniffing DNS traffic... Press Ctrl+C to stop.")
sniff(filter="udp port 53", prn=process_packet, store=0)
