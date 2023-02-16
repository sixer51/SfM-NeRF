import matplotlib.pyplot as plt
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def displayImages(images, numCol = 4):
    aspectRatio = images[0].shape[0]/images[0].shape[1]
    # plt.figure(figsize=(16/aspectRatio,16))
    numFigures = len(images)
    rowNum = numFigures//numCol + numCol
    fig = plt.figure(figsize=(numCol*64/rowNum/aspectRatio,numCol*64/numCol))
    

    ax = [fig.add_subplot(rowNum,numCol,i+1) for i in range(numFigures)]

    for i,a in enumerate(ax):
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis('off')
        a.imshow(images[i],aspect=1)

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    return