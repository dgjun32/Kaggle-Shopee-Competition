CFG_VIT = {
    'model' : {'name': 'transformers.ViTForImageClassification',
               'weight': 'google/vit-base-patch16-224',
               'img_size' : 224,
               'num_classes' : 11014,
               'scale' : 50,
               'margin' : 0.9},
    'training' : {'batch_size':64,
                  'epochs' : 10,
                  'optim' : 'torch.optim.Adam',
                  'lr_schedule':{'name':'torch.optim.lr_scheduler.CosineAnnealingLR',
                                 'learning_rate':2e-5}
                                },
    'path' : {'output' : '../model_checkpoint/image_encoder/',
              'df' : '../data/train.csv',
              'image_dir' : '../data/'
    }

CFG_BERT = {
    'model' : {'name': 'transformers.BertModel',
               'weight': 'cahya/bert-base-indonesian-522M',
               'max_length': 256,
               'num_classes':11014,
               'scale' : 50,
               'margin' : 0.7},
    'training' : {'batch_size' : 32,
                  'epochs':100,
                  'optim' : 'torch.optim.Adam',
                  'lr_schedule':{'name':'torch.optim.lr_scheduler.CosineAnnealingLR',
                                 'backbone_lr':1e-5,
                                 'arcface_lr':5e-3}
                  },
    'path' : {'output': '../model_checkpoint/text_encoder/',
              'df' : '../data/train.csv'}
    }
