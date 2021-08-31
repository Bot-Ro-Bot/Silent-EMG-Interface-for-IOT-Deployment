import socket

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

port = 1234

s.bind((socket.gethostname(), port))        
print("socket binded to %s" %(port))
 
# put the socket into listening mode
s.listen(5)    
print ("socket is listening")           
 
 
# a forever loop until we interrupt it or
# an error occurs
while True:   
    # Establish connection with client.
    c, addr = s.accept()    
    print ('Got connection from', addr )
    
    # send a thank you message to the client.
    c.send(bytes("5","utf-8"))
    
    # Close the connection with the client
    c.close()

    # sudo apt install python3-gst-1.0