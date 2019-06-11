import cv2
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

DATA_PATH = "./dataset/"

class DataLoader():
    def __init__(self, img_res=96, Train=True):
        self.img_res = img_res
        self.Train = Train

        if self.Train:
            self.train_data, self.test_data, self.train_labels, self.test_labels = self.get_train_test()

    def get_train_test(self,split_ratio=0.8, random_state=42):
        # Get available labels
        labels, indices, _ = self.get_labels(DATA_PATH)

        # Getting first arrays
        X = np.load(labels[0] + '.npy')
        y = np.zeros(X.shape[0])

        # Append all of the dataset into one single array, same goes for y
        for i, label in enumerate(labels[1:]):
            x = np.load(label + '.npy')
            X = np.vstack((X, x))
            y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

        assert X.shape[0] == len(y)

        return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)

    def get_labels(self,path=DATA_PATH):
        labels = os.listdir(path)
        label_indices = np.arange(0, len(labels))
        return labels, label_indices, to_categorical(label_indices)

    def reshape(self):
        self.train_data = self.train_data.reshape(self.train_data.shape[0], 20, 11, 1)
        self.test_data = self.test_data.reshape(self.test_data.shape[0], 20, 11, 1)

    def load_img_data(self):
        train_data=[]
        train_label=[]
        test_data=[]
        test_label=[]

        for i in range(200):
            train_label.append(int(1))
            train_data.append(cv2.imread("./jpg/Aimer/db/%03d.jpg" % (i + 1)))
        for i in range(200):
            train_label.append(int(2))
            train_data.append(cv2.imread("./jpg/Jay Chou/db/%03d.jpg" % (i + 1)))
        for i in range(200):
            train_label.append(int(3))
            train_data.append(cv2.imread("./jpg/Jolin Tsai/db/%03d.jpg" % (i + 1)))
        for i in range(200):
            train_label.append(int(4))
            train_data.append(cv2.imread("./jpg/Kenshi Yonezu/db/%03d.jpg" % (i + 1)))
        for i in range(200):
            train_label.append(int(5))
            train_data.append(cv2.imread("./jpg/Maroon5/db/%03d.jpg" % (i + 1)))

        for i in range(50):
            test_label.append(int(1))
            test_data.append(cv2.imread("./jpg/Aimer/db/%03d.jpg" % (i + 1)))
            test_label.append(int(2))
            test_data.append(cv2.imread("./jpg/Jay Chou/db/%03d.jpg" % (i + 1)))
            test_label.append(int(3))
            test_data.append(cv2.imread("./jpg/Jolin Tsai/db/%03d.jpg" % (i + 1)))
            test_label.append(int(4))
            test_data.append(cv2.imread("./jpg/Kenshi Yonezu/db/%03d.jpg" % (i + 1)))
            test_label.append(int(5))
            test_data.append(cv2.imread("./jpg/Maroon5/db/%03d.jpg" % (i + 1)))

        train_data = np.array(train_data)
        train_label = np.array(train_label)
        test_data = np.array(test_data)
        test_label = np.array(test_label)

        return train_data, train_label, test_data, test_label


    def load_batch(self, batch_size=1):
        self.n_batches = int(len(self.train_data) / batch_size)

        for i in range(self.n_batches - 1):
            batch = self.train_data[i * batch_size:(i + 1) * batch_size]
            labels = self.train_labels[i * batch_size:(i + 1) * batch_size]

            batch_label = []
            for label in labels:
                label_one_hot_encodes = self.one_hot_encode(label, num_classes=5)
                batch_label.append(label_one_hot_encodes)

            Xtr_label = np.array(batch_label)
            Xtr = np.array(batch) / 127.5 - 1.

            yield Xtr, Xtr_label

    def load_data(self, batch_size=1):
        indices = (len(self.test_labels) * np.random.rand(batch_size)).astype(int)
        batch_images = self.test_data[indices, :]
        Xte = np.array(batch_images) / 127.5 - 1.

        batch_label = []
        for label in self.test_labels[indices]:
            label_one_hot_encodes = self.one_hot_encode(label, num_classes=5)
            batch_label.append(label_one_hot_encodes)

        Xte_label = np.array(batch_label)

        return Xte, Xte_label

    def one_hot_encode(self, y, num_classes=0):
        #return np.squeeze(np.eye(num_classes)[y.reshape(-1)])
        return to_categorical(y , num_classes)

    def save_data_to_array(self,path=DATA_PATH, max_len=11):
        labels, _, _ = self.get_labels(path)

        for label in labels:
            # Init mfcc vectors
            mfcc_vectors = []

            wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
            for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
                mfcc = self.wav2mfcc(wavfile, max_len=max_len)
                mfcc_vectors.append(mfcc)
            np.save(label + '.npy', mfcc_vectors)

    def wav2mfcc(self,file_path, max_len=11):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if (max_len > mfcc.shape[1]):
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc



