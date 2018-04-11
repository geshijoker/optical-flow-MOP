import os

class settings():
  """docstring for ClassName"""
  BATCH_SIZE  = 1
  #LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
  #INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
  INITIAL_LEARNING_RATE = 1e-6
  #NUM_EPOCHS_PER_DECAY = 350
  #MOVING_AVERAGE_DECAY = 0.9999
  #NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2223 # number of images under segmentationClass file
  NUM_CLASSES = 2
  DEBUG = True
  
  MAX_STEPS = 3
  log_device_placement = False
  
  shuffle = True
  # WORKING PATH
  working_path = os.path.dirname(os.path.abspath(__file__))

  summaries_dir = os.path.join(working_path,"./summary")
  img_result = os.path.join(working_path,"./results")
  TRAIN_TXT = os.path.join(working_path,"./train_pascal.txt")
  TEST_TXT = os.path.join(working_path,'./test_pascal.txt')

  # SCRATCH STORAGE PATH
  #my_scratch = '/home/hungwei/my_scratch/'
  my_scratch="/scratch/eecs542w17_fluxg/ytchang/"
  #vgg_checkpoint_dir = os.path.join(my_scratch,'vgg_checkpoints')
  #IMAGE_DIR = os.path.join(my_scratch,'CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/JPEGImages')
  #LABEL_DIR = os.path.join(my_scratch,'CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/SegmentationClass')
  IMAGE_DIR=os.path.join(my_scratch,'data/DAVIS')
  LABEL_DIR = os.path.join(my_scratch,'CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/SegmentationClass')
  
  #IMAGE_DIR = '/home/hungwei/cv542/CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/JPEGImages'
  #LABEL_DIR = '/home/hungwei/cv542/CV_semanticSegmentation/data/TrainVal/VOCdevkit/VOC2011/SegmentationClass'
  result_dir=os.path.join(my_scratch,'data/DAVIS/Results/Segmentations/480p/542new_test_online_400/')

  # for fine_tune_lstm and train_lstm.py

  vgg_checkpoint_dir = os.path.join(my_scratch,'checkpoints/')
  #lstm_checkpoint_dir = os.path.join(my_scratch,'lstm_checkpoints/')
  #lstm_meta_graph_path = os.path.join(my_scratch,'lstm_checkpoints/lstm-model.meta')
  #lstm_meta_graph_path = os.path.join(my_scratch,'full_lstm_checkpoints/full_lstm_model.ckpt.meta')
  lstm_meta_graph_path = os.path.join(my_scratch,'full_lstm_checkpoints/lstm-model.meta')

  #run fine_tune_lstm where to save full graph checkpoint
  full_lstm_dir = os.path.join(my_scratch,'full_lstm_checkpoints/')
  full_lstm_checkpoint_path = os.path.join(full_lstm_dir,'full_lstm_model.ckpt')
  
  full_lstm_step_per_save = 1000
  
  

  #for my_test_full_lstm_pascal.py
  """
  whenever change graph, remove everything in full_lstm_dir, run fine_tune_lstm to create new checkpoint, then run train_full_lstm to train
  
  """