import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.misc import get_patient_names
from experts.fake import ExpertModel_fake


class CheXpertDataSet_rad(Dataset):
    def __init__(self, 
                image_list_file, 
                expert_model, 
                patient_names, 
                root_dir=None, 
                transform=None, 
                ignore_uncertain = True, 
                policy="ones",
                seq_len=20,
                step=1):
        """
        image_list_file: path to the file containing images with corresponding labels.
        expert_model: instance of ExpertModel_1 trained
        patient_names: which patients to include in set, dictionary
        root_dir: root directory where the images are located
        is_val: if using the validation set 
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels: ones by default
        """
        # If root_dir is not provided, try to infer it from the CSV file path
        if root_dir is None:
            # Assuming the CSV is in the same directory as the images
            root_dir = os.path.dirname(image_list_file)
        
        image_names = []
        labels = []
        rad_1 = []
        rad_2 = []
        rad_3 = []
        weights = [] # indicates if uncertainty label is present or not for each task
        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k=0
            for line in csvReader:
                k+=1

                image_name= line[0]
                # Construct the full path to the image
                full_image_path = os.path.join(root_dir, image_name)
                patient_name = line[0].split("/")[2]
                label = line[5:]
                weight = [1] *14
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1: # uncertain label
                            if ignore_uncertain:
                                weight[i] = 0
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                if patient_name in patient_names:
                    image_names.append(full_image_path)
                    labels.append(label)
                    weights.append(weight)
        exp_preds = expert_model.predict(labels)
        rad_1 = exp_preds[0]
        rad_2 = exp_preds[1]
        rad_3 = exp_preds[2]
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.rad_1 = rad_1
        self.rad_2 = rad_2
        self.rad_3 = rad_3
        self.weights = weights

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        rad1 = self.rad_1[index]
        rad2 = self.rad_2[index]
        rad3 = self.rad_3[index]
        weight = self.weights[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), torch.FloatTensor(rad1), torch.FloatTensor(rad2), torch.FloatTensor(rad3), torch.FloatTensor(weight)

    def __len__(self):
        return len(self.image_names)


imgtransCrop = 224

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(imgtransCrop),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    transforms.Resize((imgtransCrop, imgtransCrop)),
    transforms.ToTensor(),
    normalize
])

def split_dataset(train_size=0.999, random_seed=66, root_dir=None, pathFileValid=None, pathFileTrain=None, exp_fake=None, trBatchSize=16):
    dataset_all_names = get_patient_names(pathFileTrain)
    patients_train, patients_test = train_test_split(list(dataset_all_names.keys()), test_size=0.1, random_state=random_seed)
    patients_train, patients_val = train_test_split(patients_train, test_size=0.1, random_state=random_seed)
    patients_train_leftout, patients_train = train_test_split(patients_train, test_size=train_size, random_state=random_seed)

    patients_train = {p: 1 for p in patients_train}
    patients_val = {p: 1 for p in patients_val}
    patients_test = {p: 1 for p in patients_test}

    dataset_train = CheXpertDataSet_rad(pathFileTrain, exp_fake, patients_train, root_dir=root_dir, transform=transform_train)
    dataset_val = CheXpertDataSet_rad(pathFileTrain, exp_fake, patients_val, root_dir=root_dir, transform=transform_test)
    dataset_test = CheXpertDataSet_rad(pathFileTrain, exp_fake, patients_test, root_dir=root_dir, transform=transform_test)

    dataLoaderTrain = DataLoader(dataset_train, batch_size=trBatchSize, shuffle=True,  num_workers=4, pin_memory=True)
    dataLoaderVal = DataLoader(dataset_val,   batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=True)
    dataLoaderTest = DataLoader(dataset_test,  batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=True)

    dataset_all_names_val = get_patient_names(pathFileValid)
    datasetValid = CheXpertDataSet_rad(pathFileValid, exp_fake, dataset_all_names_val, transform=transform_test)
    dataLoaderOfficialValid = DataLoader(datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=0, pin_memory=True)

    print("Number of data points:")
    print(f' train: {len(dataset_train)}, test {len(dataset_test)}, val {len(dataset_val)}')

    return dataLoaderTrain, dataLoaderVal, dataLoaderTest, dataLoaderOfficialValid, patients_train_leftout