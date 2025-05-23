import socket
import atexit


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
atexit.register(sock.close)

server_address = ('EEGBot.local', 12345)  # Change to your server IP and port

# Send data

while 1:
    msg = bytearray(input(),'utf-8')
    message = b'This is the message. It will be sent in UDP.'
    sock.sendto(msg, server_address)
# Close socket
# sock.close()
