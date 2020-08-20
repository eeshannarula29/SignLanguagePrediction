import os

epochs = 2

STREAMS = 3
DIM1 = DIM2 = 64

SHAPE = shape = (DIM1,DIM2)
SHAPE_STREAMED = shape_streamed = input_shape = (DIM1,DIM2,STREAMS)

def shape_for_nsamples(n):
    return (n,DIM1,DIM2,STREAMS)

prepath = os.path.join(os.getcwd(),'data')

CATS = os.listdir(prepath)
classes = CLASSES = len(CATS)
PATHS = [os.path.join(prepath,CAT) for CAT in CATS]
