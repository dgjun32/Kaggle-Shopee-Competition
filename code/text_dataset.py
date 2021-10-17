class ShopeeTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, mode = 'train'):
        self.df = df
        self.cfg = cfg
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        text = self.df['title'][index]
        if self.mode == 'test':
            return text
        else:
            label = torch.tensor(self.df['label_group'][index]).long()
            return text, label