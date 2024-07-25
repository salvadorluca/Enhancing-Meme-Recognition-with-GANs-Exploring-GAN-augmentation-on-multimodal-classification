from torch.utils.data import Dataset

class BimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure that both embeddings and labels are of type torch.float32
        img_embedding = item['image'].float().squeeze()
        text_embedding = item['text'].float().squeeze()
        label = item['label']
        return img_embedding, text_embedding, label



       




