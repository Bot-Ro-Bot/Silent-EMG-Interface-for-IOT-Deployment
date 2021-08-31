import os
import vlc
from gtts import gTTS
from playsound import playsound

from samaye import get_time
from mausam import get_weather

import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep # Import the sleep function from the time module
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering

BATTI = 8
PANKHA = 10

SONGS = os.listdir("sangeet")
# print(SONGS)
PLAYLIST = [os.path.join("sangeet",song) for song in SONGS]
# print(PLAYLIST)
player = vlc.MediaListPlayer()

class IOT:
    def __init__(self):

        self.BATTI_FLAG = False
        self.PANKHA_FLAG = False
        GPIO.setup(BATTI, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PANKHA, GPIO.OUT, initial=GPIO.LOW)
		
        VLC = vlc.Instance("--loop")
        media_list = VLC.media_list_new()
        for song in PLAYLIST:
            media = VLC.media_new(song)
            media_list.add_media(media)
        player.set_media_list(media_list)

    def sangeet(self):
        if(player.is_playing()):
            player.pause()
            return
        player.play()

    def samaye(self):
        if(player.is_playing()):
            player.pause()
            samaye = get_time()
            self.__speak(samaye)
            player.play()
        else:
            samaye = get_time()
            self.__speak(samaye)

    def mausam(self):
        if(player.is_playing()):
            player.pause()
            mausam = get_weather()
            self.__speak(mausam)
            player.play()
        else:
            mausam = get_weather()
            self.__speak(mausam)

    def batti(self):
        self.BATTI_FLAG = not self.BATTI_FLAG
        GPIO.output(BATTI, self.BATTI_FLAG)
        print("Batti status: ",self.BATTI_FLAG)

    def pankha(self):
        self.PANKA_FLAG = not self.PANKA_FLAG
        GPIO.output(BATTI, self.PANKA_FLAG)
        print("Pankha status: ",self.PANKA_FLAG)

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
        "5": player.next,
        "6": player.previous
    }

    while True:
        prediction = input("Enter prediction: ")
        # print(prediction)
        try:
            tasks[prediction]()
        except Exception as ex:
            print("Please Enter valid prediction key or check your internet connection")
