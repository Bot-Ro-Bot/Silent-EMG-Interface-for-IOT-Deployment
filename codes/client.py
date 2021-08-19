import requests

URL = "http://127.0.0.1:5000/predict"

FILE_PATH = "../test/A.txt"

if __name__=="__main__":
    signal = open(FILE_PATH,"rb")
    values = {"file":(FILE_PATH,signal,"emg")}
    
    # request service from server through post (and sending file)
    response = requests.post(URL,files=values)
    
    # received response from server
    data = response.json()
    print("Prediction: ",data["label"])

