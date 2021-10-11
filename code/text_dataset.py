
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        tokens = self.tokenizer(self.df['title'][index])
        return tokens