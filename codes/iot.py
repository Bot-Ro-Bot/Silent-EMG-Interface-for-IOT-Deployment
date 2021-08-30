import os
import vlc
import time
from gtts import gTTS
from playsound import playsound
import random

from samaye import get_time
from mausam import get_weather

SONGS = os.listdir("sangeet")
# print(SONGS)
PLAYLIST = [os.path.join("sangeet",song) for song in SONGS]
# print(PLAYLIST)


player = vlc.MediaPlayer(PLAYLIST[random.randint(0,len(SONGS)-1)])


class IOT:
    _SANGEET_FLAG = None
    _SAMAYE_FLAG = None
    _MAUSAM_FLAG = None
    _BATTI_FLAG = None
    _PANKHA_FLAG = None

    def __init__(self):
        pass

    def sangeet(self):
        # player = vlc.MediaPlayer(PLAYLIST[0])
        if(self._SANGEET_FLAG==True):
            player.pause()
            # player = vlc.MediaPlayer(PLAYLIST[0])
            return
        val = player.play()
        if(val==0):
            self._SANGEET_FLAG = True

    def samaye(self):
        if (self._SANGEET_FLAG == True):
            player.pause()
            time.sleep(1)
        samaye = get_time()
        self.__speak(samaye)
        time.sleep(1)
        if(self._SANGEET_FLAG is not None):
            player.play()

    def mausam(self):
        if (self._SANGEET_FLAG == True):
            player.pause()
            time.sleep(1)
        mausam = get_weather()
        self.__speak(mausam)
        time.sleep(1)
        if(self._SANGEET_FLAG is not None):
            player.play()


    def batti(self):
        print("Batti balyo")

    def pankha(self):
        print("Pankha chalyo")

    def __speak(self, text):
        speak = gTTS(text=text, lang="ne", slow=False)
        file = "audio.mp3"
        speak.save(file)
        playsound(file)
        os.remove(file)


def main():
    automate = IOT()
    # automate.sangeet()
    # automate.samaye()
    # automate.mausam()


if __name__ == "__main__":
    automate = IOT()
    tasks = {
        "0": automate.samaye,
        "1": automate.sangeet,
        "2": automate.mausam,
        "3": automate.batti,
        "4": automate.pankha,
    }

    while True:
        prediction = input("Enter prediction: ")
        # print(prediction)
        try:
            tasks[prediction]()
        except Exception as ex:
            print("Please Enter valid prediction key or check your internet connection")