import  librosa
import librosa.display
#import sklearn
import matplotlib.pyplot as plt
import numpy as np
'''class mp3Tojpg():
    # 讀入音檔並提取特徵
    def __init__(self):
        dataset_name=["Jay Chou","Jolin Tsai","Kenshi Yonezu","Maroon5","Aimer"]

        for name in dataset_name:
            for i in range(1, 200):
                number = str("%03d" % i)
                filename="./dataset/" + name + "/" + number + ".mp3"
                mfccs = self.extract_feature(file=filename)
                plt.figure(figsize=(20, 5))
                librosa.display.specshow(mfccs, sr=22050, cmap='viridis')
                plt.savefig(dataset_name +'.jpg', bbox_inches='tight')
                plt.show()

def extract_feature(file):
    x, sr = librosa.load(file)
    print(len(x))
    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return norm_mfccs

if __name__ == '__main__':
    mp3Tojpg()
    extract_feature(file="./dataset/Aimer/002.mp3")
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(mfccs, sr=22050, cmap='viridis')
    plt.savefig("./jpg/Aimer/001" + '.jpg', bbox_inches='tight')
    plt.show()
'''
for i in range(200):
    y, sr = librosa.load("./dataset/Maroon5/%03d.mp3" % (i+1), duration=9)
    fig = plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y, sr=sr)
    plt.axis('off')
    plt.savefig("./jpg/Maroon5/amplitude/%03d.jpg" % (i+1), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(14, 5))
    Db = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(Db, sr=sr, x_axis='time', y_axis='linear')
    plt.axis('off')
    plt.savefig("./jpg/Maroon5/db/%03d.jpg" % (i+1), bbox_inches='tight', pad_inches=0)
    plt.close()
    del y
    print("%03d" % (i+1))
