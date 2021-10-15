
class ShopeeTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transforms, tokenizer, mode = 'train'):
        self.df = df
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        text = self.df['title'][index]
        tokens = self.tokenizer(text)
        if self.mode == 'test':
            return tokens
        else:
            label = torch.tensor(self.df['label_group'][index]).long()
            return tokens, label