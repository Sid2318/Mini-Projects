from scapy.all import sniff, DNS, DNSQR
from datetime import datetime

def process_packet(packet):
    if packet.haslayer(DNSQR):  # DNS Query only
        log = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": packet[1].src,
            "dst_ip": packet[1].dst,
            "query_domain": packet[DNSQR].qname.decode(errors="ignore"),
            "query_type": packet[DNSQR].qtype
        }
        print(f"{log['timestamp']},{log['src_ip']},{log['dst_ip']},{log['query_domain']},{log['query_type']}")

print("Sniffing DNS traffic... Press Ctrl+C to stop.")
sniff(prn=process_packet, store=0)  # no filter here
# “Give me all the packets. I’ll figure out if it’s a DNS query myself.”