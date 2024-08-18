from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on STL-10 dataset")
    parser.add_argument('--data_path', type=str, default="/mnt/d/OneDrive - Oklahoma A and M System/RA/Fall 23/Codes/MPL/STL10/",
                        help="Path to the dataset")
    parser.add_argument('-sz', '--resize', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for training and validation")
    parser.add_argument('--num_epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument('--model_checkpoint', type=str, default='./model/bywl_unsup.pt',
                        help="Path to the pre-trained model checkpoint")
    parser.add_argument('--save_path', type=str, default='./model/downstream.pt',
                        help="Directory to save the trained model and optimizer state")

    return parser.parse_args()

def main():
    args = parse_args()

    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.ToTensor(),])

    TRAIN_DATASET = torchvision.datasets.ImageFolder(root=args.data_path+'train/', transform=TRANSFORM_IMG)
    TEST_DATASET = torchvision.datasets.ImageFolder(root=args.data_path+'test/', transform=TRANSFORM_IMG)

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Determine the number of classes dynamically
    num_classes = len(TRAIN_DATASET.classes)

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    state = torch.load(args.model_checkpoint)
    model.load_state_dict(state['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(state['optimizer_state_dict'])

    best_accuracy = 0
    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0    
        
        for inputs, labels in train_loader:

            # Forward pass
            outputs = model(inputs.cuda())
        
            # Compute the loss
            loss = criterion(outputs, labels.cuda())
        
            # Zero gradients, backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Compute average training loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {avg_loss:.4f}")
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            
            loop = 0
            for images, labels in val_loader:

                # Move data to the GPU if available
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
        
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                loop += 1

                if epoch == args.num_epochs-1:
                    continue
                elif loop == args.num_epochs-1:
                    break

        # Compute validation accuracy for the epoch
        val_accuracy = 100 * total_correct / total_samples
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy      
        
            # Save the model and optimizer state if accuracy improves
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.save_path}")

if __name__ == '__main__':
    main()