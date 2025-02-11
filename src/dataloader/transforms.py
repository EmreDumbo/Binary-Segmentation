from torchvision import transforms

class Transforms:
    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    @staticmethod
    def get_test_transforms():  
        return transforms.Compose([
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])