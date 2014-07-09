import numpy as np


class TrainSubject():
    """A class to hold MEG data about a single training subject"""
    def __init__(self, idnum, data_path="", timesel=Ellipsis, colsel=Ellipsis):
        self.idnum = idnum
        npz = np.load(data_path + 'train_subject{0:0>2}.npz'.format(idnum))
        self.X = npz['X'][:, timesel, colsel]
        self.y = npz['y']
        self.X_s = self.X[self.y == 0]
        self.X_f = self.X[self.y == 1]


class TestSubject():
    """A class to hold MEG data about a single test subject"""
    def __init__(self, idnum, data_path="", timesel=Ellipsis, colsel=Ellipsis):
        self.idnum = idnum
        npz = np.load(data_path + 'test_subject{0:0>2}.npz'.format(idnum))
        self.X = npz['X'][:, timesel, colsel]
