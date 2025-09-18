from scapy.all import sniff, DNS, IP, IPv6, UDP, conf
import time
import json
import datetime

class DNSCapture:
    def __init__(self, interface=None):
        self.interface = interface
        self.running = True
        self.output_file = 'dns_capture.txt'
        
        # Set interface if provided
        if interface:
            conf.iface = interface
            
        # Initialize output file with header
        self.init_output_file()
    
    def init_output_file(self):
        """Initialize the output file with header"""
        with open(self.output_file, 'w') as f:
            f.write("Timestamp,Source_IP,Destination_IP,Query_Domain,Query_Type\n")
    
    def process(self):
        """Main capture loop"""
        print(f"Starting DNS capture on interface: {self.interface or 'default'}")
        print("Scanning for DNS requests... Press Ctrl+C to stop")
        
        while self.running:
            try:
                sniff(filter="udp and port 53", prn=self.check_packet, count=1, store=0)
            except KeyboardInterrupt:
                print("\nStopping capture...")
                self.running = False
                break
            except Exception as e:
                print(f"Error during capture: {e}")
                break
    
    def check_packet(self, pkt):
        """Process individual DNS packets"""
        try:
            # Get source and destination IPs (support both IPv4 and IPv6)
            src_ip = None
            dst_ip = None
            
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
            elif IPv6 in pkt:
                src_ip = pkt[IPv6].src
                dst_ip = pkt[IPv6].dst
            else:
                return  # Skip if no IP layer
            
            # Process DNS layer
            if DNS in pkt:
                dns_layer = pkt[DNS]
                
                # Only process queries (not responses)
                if dns_layer.qr == 0 and dns_layer.qd:  # Query with question section
                    query_domain = dns_layer.qd.qname.decode('utf-8').rstrip('.')
                    query_type = self.get_query_type(dns_layer.qd.qtype)
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Create data record
                    data = {
                        "timestamp": timestamp,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "query_domain": query_domain,
                        "query_type": query_type
                    }
                    
                    # Write to file and display
                    self.write_to_file(data)
                    print(f"Captured: {query_domain} ({query_type}) from {src_ip} to {dst_ip}")
                    
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def get_query_type(self, qtype):
        """Convert query type code to readable format"""
        query_types = {
            1: 'A',       # IPv4 address
            2: 'NS',      # Name server
            5: 'CNAME',   # Canonical name
            6: 'SOA',     # Start of authority
            12: 'PTR',    # Pointer record
            15: 'MX',     # Mail exchange
            16: 'TXT',    # Text record
            28: 'AAAA',   # IPv6 address
            33: 'SRV',    # Service record
            65: 'HTTPS'   # HTTPS record
        }
        return query_types.get(qtype, f'Type_{qtype}')
    
    def write_to_file(self, data):
        """Write DNS data to file"""
        with open(self.output_file, 'a') as f:
            f.write(f"{data['timestamp']},{data['src_ip']},{data['dst_ip']},{data['query_domain']},{data['query_type']}\n")

def main():
    print("DNS Traffic Capture Tool")
    print("=" * 40)
    
    # Create DNS capture instance
    dns_capture = DNSCapture()
    
    try:
        # Start capture process
        dns_capture.process()
        
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"DNS queries saved to {dns_capture.output_file}")

if __name__ == "__main__":
    main()