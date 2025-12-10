from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
import os
import re
import torch
import cv2


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


# --------------------------------------------------------------------------------
# DISPLAY A GRID OF RANDOM IMAGES FROM A DATASET INSTANCE
# INPUTS: 
#     - dataset (Dataset): Instance of Dataset
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
# --------------------------------------------------------------------------------
def showDataset(dataset, param):

    rows, cols= 5, 3
    numPairs = rows*cols

    indices = random.sample(range(len(dataset)), numPairs)

    fig, axes = plt.subplots(rows, cols*2, figsize=(6,6), dpi=150)

    cmap = mcolors.ListedColormap([
        "black",   # 0 = "other"
        "blue",    # 1 = "water"
        "red",     # 2 = "building"
        "yellow",  # 3 = "farmland"
        "green"    # 4 = "forest"
    ])

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 2)

    for i, idx in enumerate(indices):
        image, mask, tileName, resizedImg = dataset[idx]

        row = i // cols
        col = i % cols

        # left subplot: image
        axes[row, col*2].imshow(resizedImg.numpy().astype('uint8').transpose(1,2,0))
        axes[row, col*2].set_title(tileName, fontsize=6)
        axes[row, col*2].axis('off')

        # right subplot: mask
        axes[row, col*2 + 1].imshow(mask.numpy().astype('uint8'), cmap=cmap, vmin=0, vmax=4)
        axes[row, col*2 + 1].set_title("mask", fontsize=6)
        axes[row, col*2 + 1].axis('off')

    plt.tight_layout
    plt.show()


# --------------------------------------------------------------------------------
# Reconstruction of full images from tiles and save results
# Inputs:
#     - allResizedImgs (np.ndarray): array of image tiles with shape (N, 3, H, W)
#     - allMasksPreds  (np.ndarray): array of predicted mask tiles with shape (N, H, W)
#     - allMasks       (np.ndarray): array of ground truth mask tiles with shape (N, H, W)
#     - allTileNames   (list of str): list of tile filenames, formatted as "tile_xx_yy.png"
# --------------------------------------------------------------------------------
def reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, allClustersPreds, savePath):
   
    # Parse tile coordinates from filenames
    coords = []
    for fname in allTileNames:
        m = re.search(r"tile_(\d+)_(\d+)\.png", fname)
        if not m:
            raise ValueError(f"Filename {fname} does not match expected pattern")
        xx, yy = map(int, m.groups())
        coords.append((xx, yy))
    
    coords = np.array(coords)
    minX, minY = coords.min(axis=0)
    maxX, maxY = coords.max(axis=0)
    
    # Determine tile size and full image dimensions
    tileH, tileW = allResizedImgs.shape[2], allResizedImgs.shape[3]
    fullH = (maxX - minX + 1) * tileH
    fullW = (maxY - minY + 1) * tileW
    
    # Allocate empty arrays for reconstruction
    fullImg  = np.zeros((3, fullH, fullW), dtype=allResizedImgs.dtype)
    fullPredMask = np.zeros((3, fullH, fullW), dtype=np.uint8)
    fullPred = np.zeros((3, fullH, fullW), dtype=allClustersPreds.dtype)
    fullMask = np.zeros((3, fullH, fullW), dtype=allMasks.dtype)
    # Stitch image, predicted mask, and ground truth mask tiles
    for imgTile, predTile, maskTile, (xx, yy) in zip(allResizedImgs, allClustersPreds, allMasksPreds, coords):
        i = xx - minX
        j = yy - minY
        y0, y1 = i * tileH, (i + 1) * tileH
        x0, x1 = j * tileW, (j + 1) * tileW
        
        fullImg[:, y0:y1, x0:x1] = imgTile
        fullPred[:, y0:y1, x0:x1]   = predTile
        fullPredMask[:, y0:y1, x0:x1]   = maskTile

    # Create overlay of predicted mask on original image
    imgRGB   = np.transpose(fullImg, (1, 2, 0))
    imgProxy = np.transpose(fullPredMask, (1, 2, 0))
    imgPred  = np.transpose(fullPred, (1, 2, 0))
    overlay = imgRGB.copy()
    
    cmap = mcolors.ListedColormap([
        "black",   # 0 = "other"
        "blue",    # 1 = "water"
        "red",     # 2 = "building"
        "yellow",  # 3 = "farmland"
        "green"    # 4 = "forest"
    ])
    
    predColors = cmap(imgPred.mean(axis=2))[..., :3]
    alpha = 0.4
    overlay = (1 - alpha) * imgRGB / 255.0 + alpha * predColors

    # Save results
    createFolder(savePath)
    plt.imsave(os.path.join(savePath, "image.png"), imgRGB)
    plt.imsave(os.path.join(savePath, "image_proxy.png"), imgProxy)
    plt.imsave(os.path.join(savePath, "prediction.png"), imgPred, cmap=cmap)
    plt.imsave(os.path.join(savePath, "overlay.png"), overlay)





    
    
def showPredictions(dataset, model, device, param, rows=4, cols=3):
    """
    Display a grid of random predictions:
    [image | ground truth | predicted mask]

    dataset : e.g. myNetwork.dataSetTest
    model   : e.g. myNetwork.model
    device  : e.g. myNetwork.device ("cuda" or "cpu")
    """

    model.eval()

    numPairs = rows * cols
    indices = random.sample(range(len(dataset)), numPairs)

    fig, axes = plt.subplots(rows, cols * 3, figsize=(9, 6), dpi=150)

    cmap = mcolors.ListedColormap([
        "black",   # 0 = "other"
        "blue",    # 1 = "water"
        "red",     # 2 = "building"
        "yellow",  # 3 = "farmland"
        "green"    # 4 = "forest"
    ])

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 3)

    for i, idx in enumerate(indices):
        # Get a sample: image (tensor), mask (tensor), tileName, resizedImg (tensor)
        image, mask, tileName, resizedImg = dataset[idx]

        # Prepare input for model: add batch dimension, send to device
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)            # (1, C, H, W)
            pred = torch.argmax(logits, dim=1)      # (1, H, W)
            pred = pred.squeeze(0).cpu()            # (H, W)

        row = i // cols
        col = i % cols

        # ---- Left: original image (resized) ----
        axes[row, col*3].imshow(
            resizedImg.numpy().astype('uint8').transpose(1, 2, 0)
        )
        axes[row, col*3].set_title(tileName, fontsize=6)
        axes[row, col*3].axis('off')

        # ---- Middle: ground truth mask ----
        axes[row, col*3 + 1].imshow(
            mask.numpy().astype('uint8'),
            cmap=cmap, vmin=0, vmax=4
        )
        axes[row, col*3 + 1].set_title("GT", fontsize=6)
        axes[row, col*3 + 1].axis('off')

        # ---- Right: predicted mask ----
        axes[row, col*3 + 2].imshow(
            pred.numpy().astype('uint8'),
            cmap=cmap, vmin=0, vmax=4
        )
        axes[row, col*3 + 2].set_title("Pred", fontsize=6)
        axes[row, col*3 + 2].axis('off')

    plt.tight_layout()
    plt.show()


def postprocess_prediction(fullPred):
    """
    fullPred: (H, W) numpy array of class indices (prediction).

    Class IDs (from the table):
      0 -> Others
      1 -> Water
      2 -> Buildings
      3 -> Farmlands
      4 -> Green Spaces
    """
    fullPred_pp = fullPred.copy().astype(np.int32)

    kernel = np.ones((3, 3), np.uint8)

    # 1) Smooth / fill Farmlands (3) and Green Spaces (4)
    for cls in (3, 4):
        bin_mask = (fullPred == cls).astype(np.uint8)

        # Closing: dilation then erosion -> fills small holes
        closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Where closed == 1, set class to cls
        fullPred_pp[closed == 1] = cls

    # 2) Remove tiny Building blobs (2)
    cls = 2
    bin_mask = (fullPred_pp == cls).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    min_area = 30  # tune this: smaller => more tiny dots removed
    for lab in range(1, num_labels):  # 0 is background
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < min_area:
            fullPred_pp[labels == lab] = 0  # send tiny blobs to "Others" (0)

    return fullPred_pp.astype(fullPred.dtype)
