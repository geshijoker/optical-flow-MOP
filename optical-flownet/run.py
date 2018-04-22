from train import Trainer
from config import *
import input_data as inp


def main(argv=None):
    gpu_batch_size = int(batch_size)
    devices = ['/gpu:0']
    data_dic, labels_dic = read_data_label_dir("/ImageSets/480p/train.txt")
    train_data = input_raw(data_dic['bear'])
    tr = Trainer(train_data, None)

if __name__ == '__main__':
    tf.app.run()