import os
import datetime

SONGS = os.listdir("songs")
os.chdir("songs")

class IOT:
    _SANGEET_FLAG = None
    _SAMAYE_FLAG = None
    _MAUSAM_FLAG = None
    _BATTI_FLAG = None
    _PANKHA_FLAG = None

    def __init__(self):
        pass

    def sangeet(self):
        cmd = "setsid vlc "+ SONGS[0]
        # print(cmd)
        try :
            os.system(cmd)
        except Exception as ex:
            print("Cannot play music because ",ex)
   
    def samaye(self):
        time = datetime.datetime.now()
        print(time)
  
    def mausam(self):
        pass
   
    def batti(self):
        pass

    def pankha(self):
        pass



def main():
    automate = IOT()
    # automate.sangeet()
    automate.samaye()

if __name__ == "__main__":
    main()
    # print(SONGS)