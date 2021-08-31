import socket

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

port = 12345

s.connect((socket.gethostname(),port))
prediction = s.recv(1024)
print(prediction.decode("utf-8"))

s.close()
