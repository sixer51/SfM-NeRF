# %%
from DataLoader import *
from Visualization import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from GetInliersRANSAC import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
import os

# Don't generate pyc codes
sys.dont_write_bytecode = True

# %% [markdown]
# 

# %%
dataDirPath = os.getcwd() + '/P3Data/'
images = loadImages(dataDirPath)
# displayImages(images, 5)


# %%
K = getCameraParams(dataDirPath)
print(K)

# %%
featureMatchesList = parseMatchFiles(dataDirPath)

# %%
InlierList, OutlierList = ransac(featureMatchesList[0].matchPairs, 1000, 0.02)
matchPairsInliers = [featureMatchesList[0].matchPairs[i] for i in InlierList]
matchPairsOutliers = [featureMatchesList[0].matchPairs[i] for i in OutlierList]


# %%
matchImage = drawMatchs(images[0], images[1], matchPairsInliers)

plt.figure(figsize=(16, 8))
plt.axis("off")
plt.imshow(matchImage)
plt.show()

# %% [markdown]
# ## Final Fundimental Matrix Calculated from inliers

# %%
x1s = []
x2s = []
for idx in range(len(matchPairsInliers)):
    x1s.append(matchPairsInliers[idx].coords1)
    x2s.append(matchPairsInliers[idx].coords2)
x1s = np.array(x1s)
x2s = np.array(x2s)

Fest = EstimateFundamentalMatrix(x1s, x2s)
print(Fest)

# %% [markdown]
# ## Estimate Essential Matrix from Fundimental Matrix

# %%
Eest = EssentialFromFundamental(Fest, K)
print(Eest)

# %% [markdown]
# ## Estimate Camera Pose

# %%
Rest, Test = extractCameraPose(Eest)
print(Rest)
print(Test)

# %% [markdown]
# ## Triangulation Check for Cheirality Condition

# %%
Xest = []
for i in range(len(Rest)):
    X = LinearTriangulation(K, x1s, x2s, np.eye(3), np.zeros((3,1)), Rest[i], Test[i])
    Xest.append(X)

Rest, Test, Xest = disambiguateCameraPose(Rest, Test, Xest)


# %%
drawWorldPoints([Xest])
# drawWorldPoints([Xest[0]])
# drawWorldPoints([Xest[1]])
# drawWorldPoints([Xest[2]])
# drawWorldPoints([Xest[3]])


