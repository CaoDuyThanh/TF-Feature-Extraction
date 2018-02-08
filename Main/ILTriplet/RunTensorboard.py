tensorBoardPath = '/media/badapple/Data/Projects/Machine\ Learning/TF-Feature-Extraction/Pretrained/ILTriplet/Raw/02369-14578/Tensorboard/TripletAE/'

def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()