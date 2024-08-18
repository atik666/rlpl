import torch
from bywl_main import BYWL
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
import argparse
import os
import torch
from torchvision import models
from torch import nn

# Initialize the parser
parser = argparse.ArgumentParser(
    description="Representation Learning with BYWL."
    )

# Add the parameters
parser.add_argument('-p', '--path', type=str, 
                    default='/mnt/d/OneDrive - Oklahoma A and M System/RA/Fall 23/Codes/MPL/STL10/', 
                    help='Path to the dataset')
parser.add_argument('-sz', '--resize', type=int, default=256, help='Image size')
parser.add_argument('-b', '--batchsz', type=int, default=128, help='Batch size')
parser.add_argument('-s', '--save', type=str, default='./model/', help='Path to save the models')
parser.add_argument('-sp', '--save_path', type=str, default='./model/bywl_unsup.pt', help='file name')
parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of  epochs')
parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('-m', '--method', type=str, default='BYWL', help='Methods to perform', choices=['BYWL', 'BYOL'])

# Parse the arguments
args = parser.parse_args()
print(args)

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize([args.resize, args.resize]),
    transforms.ToTensor(),])

try:
    TRAIN_UNLABELED_DATASET = torchvision.datasets.ImageFolder(root=args.path+'unsup/', transform=TRANSFORM_IMG)
except:
    print("Make sure to put the unlableled data in the 'unsup' folder. Or change the dir name.")

TRAIN_DATASET = torchvision.datasets.ImageFolder(root=args.path+'train/', transform=TRANSFORM_IMG)
# Get the number of classes
num_classes = len(TRAIN_DATASET.classes)

# Concatenate the two datasets into one
# Combine labelled and unlabled data to learn representation.
combined_dataset = ConcatDataset([TRAIN_UNLABELED_DATASET, TRAIN_DATASET]) 

unlbl_loader = DataLoader(
    combined_dataset,
    batch_size=args.batchsz,
    shuffle=True,
    drop_last=True,
)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

if os.path.exists(args.save_path):
    state = torch.load(args.save_path)
    model.load_state_dict(state['model_state_dict'])
    print("model loaded from the checkpoint. \n")
    print("least loss: ", state['loss'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

learner = BYWL(
    model.to(device),
    image_size = args.resize,
    method = args.method,
    hidden_layer = 'avgpool'
)

opt_learner = torch.optim.AdamW(learner.parameters(), lr=args.lr)
opt_res = torch.optim.AdamW(model.parameters(), lr=args.lr)

# If save dir does not exist, create it.
if not os.path.exists(args.save):
    os.makedirs(args.save)

if os.path.exists(args.save_path):
    opt_res.load_state_dict(state['optimizer_state_dict'])
    print("optimizer loaded from the checkpoint. \n")

least_loss = float('inf')
num_epochs = args.epochs
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    total_loss = 0  
    for images, _ in tqdm(unlbl_loader):
        loss = learner(images.to(device))
        opt_learner.zero_grad()
        opt_res.zero_grad()
        loss.backward()
        opt_learner.step()
        opt_res.step()
        learner.update_moving_average() # update moving average of target encoder
        total_loss += loss.item()
    # Compute average training loss for the epoch
    avg_loss = total_loss / len(unlbl_loader)
    print(f"\n Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

    # Check if the current model has the best accuracy
    if avg_loss < least_loss:
        least_loss = avg_loss

        # Save model parameters along with validation loss information
        state = {
            'model_state_dict': model.state_dict(),
            'loss': least_loss,
            'optimizer_state_dict': opt_res.state_dict()
        }

        # save your improved network
        torch.save(state, args.save+'bywl_unsup_best_loss.pt')
        print("model saved! \n")

state = {
    'model_state_dict': model.state_dict(),
    'loss': least_loss,
    'optimizer_state_dict': opt_res.state_dict()
}

# save your last network
torch.save(state, args.save_path)
print("Final model saved!")