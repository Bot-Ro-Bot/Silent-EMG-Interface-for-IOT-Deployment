import os
from gtts import gTTS
from playsound import playsound

from samaye import get_time
from mausam import get_weather


SONGS = os.listdir("sangeet")
# os.chdir("songs")


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
        samaye = get_time()
        self.__speak(samaye)

    def mausam(self):
        mausam = get_weather()
        self.__speak(mausam)

    def batti(self):
        pass

    def pankha(self):
        pass

    def __speak(self,text):
        speak = gTTS(text=text, lang="ne", slow=False)
        file = "audio.mp3"
        speak.save(file)
        playsound(file)
        os.remove(file)

def main():
    automate = IOT()
    # automate.sangeet()
    automate.samaye()
    automate.mausam()

if __name__ == "__main__":
    main()
    print("Done !!")
    # print(SONGS)