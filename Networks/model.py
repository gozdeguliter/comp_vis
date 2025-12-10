from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import BasicNet
from Networks.Architectures.unet import UNet

import numpy as np
np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
import torch.nn.functional as F
import cv2
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]   # "cuda" or "cpu"
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = UNet(param).to(self.device)
        #self.model = BasicNet(param).to(self.device)

        # -------------------
        # TRAINING PARAMETERS
        # -------------------

        # Class IDs: [0,    1,     2,       3,        4]
        #           other  water building farmland  green
        class_weights = torch.tensor(
            [1.2,  1.3,   1.05,   0.8,     1.5],   # tweakable!
            dtype=torch.float32,
            device=self.device
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

                
        
        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain = LLNDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal   = LLNDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest  = LLNDataset(imgDirectory, maskDirectory, "test",  param)

        self.trainDataLoader = DataLoader(self.dataSetTrain,
                                          batch_size=self.batchSize,
                                          shuffle=True,
                                          num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,
                                          batch_size=self.batchSize,
                                          shuffle=False,
                                          num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,
                                          batch_size=self.batchSize,
                                          shuffle=False,
                                          num_workers=4)

    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS 
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        wghtsPath = os.path.join(self.resultsPath, "_Weights", "wghts.pkl")
        if os.path.exists(wghtsPath):
            state_dict = torch.load(wghtsPath, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            print(f"Loaded weights from {wghtsPath}")
        else:
            print(f"⚠️ No weights found at {wghtsPath}. Train the model first.")

    # -----------------------------------
    # TRAINING LOOP
    # -----------------------------------


        # -----------------------------------
    # TRAINING LOOP
    # -----------------------------------
    def train(self):

        # Histories for plots
        train_loss_history = []
        val_loss_history   = []

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())

        for i in range(self.epoch):
            # -------------------
            # TRAINING PHASE
            # -------------------
            self.model.train(True)
            running_train_loss = 0.0
            running_train_correct = 0
            running_train_total   = 0

            for (images, masks, tileNames, resizedImgs) in self.trainDataLoader:

                images = images.to(self.device, dtype=torch.float)
                masks  = masks.to(self.device, dtype=torch.float)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(images)        # (B, C, H, W)

                # Loss
                loss = self.criterion(outputs, masks)    
                
                # Backprop + update
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(self.trainDataLoader)

            train_loss_history.append(epoch_train_loss)

            # -------------------
            # VALIDATION PHASE
            # -------------------
            self.model.eval()
            running_val_loss = 0.0
            running_val_correct = 0
            running_val_total   = 0

            with torch.no_grad():
                for (images, masks, tileNames, resizedImgs) in self.valDataLoader:
                    images = images.to(self.device, dtype=torch.float)
                    masks  = masks.to(self.device, dtype=torch.float)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks.long())
                    running_val_loss += loss.item()

                    # ---- VAL ACCURACY ----

            epoch_val_loss = running_val_loss / len(self.valDataLoader)

            val_loss_history.append(epoch_val_loss)

            print(
                f"Epoch {i+1}/{self.epoch} - "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}, "
            )

            # Save best weights based on val loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

        # -----------------------------------
        # AFTER TRAINING: SAVE CURVES & WEIGHTS
        # -----------------------------------
        graphsPath = os.path.join(self.resultsPath, "_Graphs")
        createFolder(graphsPath)

        # 1. Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True)
        curvesPath = os.path.join(graphsPath, "training_curves.png")
        plt.savefig(curvesPath, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved at {curvesPath}")

        # 3. Best model weights
        wghtsPath = os.path.join(self.resultsPath, "_Weights")
        createFolder(wghtsPath)
        torch.save(best_model_wts, os.path.join(wghtsPath, "wghts.pkl"))
        print(f"Best model weights saved at {os.path.join(wghtsPath, 'wghts.pkl')}")




        # -------------------------------------------------
    # EVALUATION PROCEDURE
    # -------------------------------------------------
    def evaluate(self):
        self.model.train(False)
        self.model.eval()
         
        allMasks, allMasksPreds, allTileNames, allResizedImgs, allClustersPreds = [], [], [], [], []
        tensorMasks = []
        for (images, masks, tileNames, resizedImgs) in self.testDataLoader:
            images = images.to(self.device, dtype=torch.float)
            masks  = masks.to(self.device, dtype=torch.float)

            outputs = self.model(images)

            images = images.to('cpu')
            masks  = masks.to('cpu')
            outputs = outputs.to('cpu')

            norm = alb.Compose([alb.Normalize(mean=(0.485, 0.456, 0.406), 
                                 std=(0.229, 0.224, 0.225), 
                                 normalization="min_max")])
        
            
            outputs_norm = norm(image=outputs.detach().numpy())

            clustersPreds = self.produce_clusters(outputs_norm["image"])
            masksPreds = outputs_norm["image"] * 255.0 

            allMasks.extend(masks.numpy())
            tensorMasks.extend(masks)
            allMasksPreds.extend(masksPreds)
            allResizedImgs.extend(resizedImgs.numpy())
            allClustersPreds.extend(clustersPreds)
            allTileNames.extend(tileNames)
        
        allMasks       = np.array(allMasks)        # (N,H,W)
        allMasksPreds  = np.array(allMasksPreds)   # (N,H,W)
        allResizedImgs = np.array(allResizedImgs)
        allClustersPreds = np.array(allClustersPreds)


        # -----------------------------
        # QUALITATIVE EVALUATION
        # -----------------------------
        savePath = os.path.join(self.resultsPath, "Test")
        createFolder(savePath)
        reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, allClustersPreds, savePath)
        print(f"Qualitative results saved in {savePath}")


    def produce_clusters(self, imgs):

        clustered_imgs = []

        for img in imgs:
            pixel_vals = img.reshape((-1,3))

            pixel_vals = np.float32(pixel_vals)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

            k = 5
            retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]

            #print(labels.shape)
            #print(centers.shape)


            segmented_image = segmented_data.reshape(img.shape)

            clustered_imgs.append(segmented_image)

        return clustered_imgs

def dice_loss_multiclass(logits, targets, eps=1e-6):
    """
    logits: (B, C, H, W) raw scores from the network
    targets: (B, H, W) integer class labels
    """
    num_classes = logits.shape[1]

    # Probabilities
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # One-hot encode targets
    targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()         # (B, C, H, W)

    # Flatten
    probs_flat   = probs.contiguous().view(probs.shape[0], num_classes, -1)
    targets_flat = targets_one_hot.contiguous().view(targets_one_hot.shape[0], num_classes, -1)

    intersection = (probs_flat * targets_flat).sum(-1)          # (B, C)
    union        = probs_flat.sum(-1) + targets_flat.sum(-1)    # (B, C)

    dice = (2.0 * intersection + eps) / (union + eps)           # (B, C)
    dice_loss = 1.0 - dice.mean()                               # scalar

    return dice_loss


