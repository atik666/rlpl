import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TFMPL for pseudo labeling on the downstream tasks")
    parser.add_argument('--data_path', type=str, default='/mnt/d/OneDrive - Oklahoma A and M System/RA/Fall 23/Codes/MPL/STL10/',
                        help="Path to the dataset")
    parser.add_argument('--resize', type=int, default=256, help="Resize dimensions for input images")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation")
    parser.add_argument('--unlbl_batch_size', type=int, default=128, help="Batch size for unlabeled data")
    parser.add_argument('--num_epochs', type=int, default=151, help="Number of training epochs")
    parser.add_argument('--threshold', type=float, default=0.95, help="Threshold for pseudo-labeling")
    parser.add_argument('--model_path', type=str, default='./model/downstream.pt', help="Path to the pre-trained model")
    parser.add_argument('--save_path', type=str, default='./model/final.pt', help="Directory to save the trained models")
    parser.add_argument('--device_ids', type=list, default=[0, 1], help="List of GPU device IDs for DataParallel")
    parser.add_argument('--prob', type=float, default=0.25, help="Probability of image weak augmentation")
    return parser.parse_args()

def main():
    args = parse_args()

    # Data transformations
    transform_img = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.ToTensor(),
    ])

    weak_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([args.resize, args.resize]),
        transforms.RandomHorizontalFlip(p=args.prob),
        transforms.RandomVerticalFlip(p=args.prob),
        transforms.ToTensor(),
    ])

    # Load datasets
    TRAIN_DATASET = torchvision.datasets.ImageFolder(root=args.data_path+'train/', transform=transform_img)
    TEST_DATASET = torchvision.datasets.ImageFolder(root=args.data_path+'test/', transform=transform_img)
    TRAIN_UNLABELED_DATASET = torchvision.datasets.ImageFolder(root=args.data_path+'unsup/', transform=transform_img)

    # Data loaders
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

    unlbl_loader = DataLoader(
        TRAIN_UNLABELED_DATASET,
        batch_size=args.unlbl_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    # Determine the number of classes dynamically
    num_classes = len(TRAIN_DATASET.classes)

    # Initialize teacher and student models
    teacher = resnet18(weights=None)
    teacher.fc = nn.Linear(512, num_classes)  # Assuming 10 classes, change if necessary

    student = resnet18(weights=None)
    student.fc = nn.Linear(512, num_classes)  # Assuming 10 classes, change if necessary

    # Load pre-trained teacher model
    state = torch.load(args.model_path)
    teacher.load_state_dict(state['model_state_dict'])
    student.load_state_dict(state['model_state_dict'])

    teacher = teacher.to(device)
    teacher = torch.nn.DataParallel(teacher, device_ids=args.device_ids)

    student = student.to(device)
    student = torch.nn.DataParallel(student, device_ids=args.device_ids)

    # Optimizers
    optimizer_teacher = torch.optim.SGD(teacher.parameters(), lr=3e-4, momentum=0.9)
    optimizer_student = torch.optim.SGD(student.parameters(), lr=3e-4, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    def model_eval(model, name: str):
        """Evaluate the model on the validation set."""
         
        model.eval()  # Set the model to evaluation mode
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to the GPU if available
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
        
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        # Compute validation accuracy for the epoch
        val_accuracy = 100 * total_correct / total_samples
        print(f"{name} validation accuracy: {val_accuracy:.2f}%")

        return val_accuracy

    model_eval(student, name = "Student")
    model_eval(teacher, name = "Teacher")

    best_accuracy = 0.0
    print("Training started with threshold:", args.threshold)

    for epoch in range(args.num_epochs):

        teacher.train()
        student.train()  

        for i, (labeled_data, unlabeled_data) in tqdm(enumerate(zip(train_loader, unlbl_loader)), total=len(train_loader)+len(unlbl_loader)):

            inputs, labels = labeled_data[0].to(device), labeled_data[1].to(device)
            inputs_unlabel = unlabeled_data[0].to(device)

            optimizer_teacher.zero_grad()
            optimizer_student.zero_grad()

            # Forward pass through student
            with torch.no_grad():       # no gradient here
                y_preds = student(inputs)
                loss_st_1 = criterion(y_preds, labels)  # first loss come from student without gradient to reduce computation MPL

            # Pseudo-labeling with teacher
            y_unlabel_preds_te = teacher(inputs_unlabel) # TODO: Maybe do max vote here

            # Applying pseudo labeling in teacher with a threshold
            t_t = F.softmax(y_unlabel_preds_te.detach(), dim=1)
            t_t_max, pseudo_lbl = torch.max(t_t, dim=1)
            pseudo_y = pseudo_lbl[t_t_max >= args.threshold]    # TODO: need to improve the number of pseudo_y
            y_unlabel_preds_te_n = y_unlabel_preds_te[t_t_max >= args.threshold] # taking log_pred of teacher after threshold filtering

            inputs_aug_student = torch.stack([weak_transform(tensor) for tensor in inputs_unlabel]) # TODO: weak augmentation # Here only taken threshold and vote filtered inputs
            y_unlabel_preds_st = student(inputs_aug_student.to(device))  # TODO: weak augmentation # Here only taken threshold and vote filtered inputs
            y_unlabel_preds_st = y_unlabel_preds_st[t_t_max >= args.threshold]

            loss_st_unlabel = criterion(y_unlabel_preds_st, pseudo_y) # loss from student log prediction and teacher threshold and vote filtered labels
                        
            loss_st_unlabel.backward()
            optimizer_student.step()

            with torch.no_grad():
                y_preds = student(inputs) # original labelled inputs
                loss_st_2 = criterion(y_preds, labels) # loss on labelled data from student

            change = loss_st_2 - loss_st_1 

            mlp_loss = change * criterion(y_unlabel_preds_te_n, pseudo_y) # loss for MPL

            y_preds = teacher(inputs)  # original labelled inputs
            loss_te = criterion(y_preds, labels) # loss on labelled data from teacher

            loss = loss_te + mlp_loss # Total loss
            loss.backward()
            optimizer_teacher.step()
        
        print(f"Epoch [{epoch + 1}/{args.num_epochs}]")

        if epoch % 10 == 0:
            model_eval(student, name = "Student")
            val_accuracy = model_eval(teacher, name = "Teacher")

            # Check if the current model has the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                state = {
                    'model_state_dict': teacher.state_dict(),
                    'acc': best_accuracy,
                    'optimizer_state_dict': optimizer_teacher.state_dict()
                }

                # Save the model
                torch.save(state, args.save_path)
                print("Model saved")

if __name__ == '__main__':
    main()