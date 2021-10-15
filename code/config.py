CFG_VIT = {
    'model' : {'name': 'transformers.ViTForImageClassification',
               'weight': 'google/vit-base-patch16-224',
               'img_size' : 224,
               'num_classes' : 11014,
               'scale' : 50,
               'margin' : 0.4},
    'training' : {'batch_size':64,
                  'epochs' : 10,
                  'optim' : 'torch.optim.Adam',
                  'lr_schedule':{'name':'torch.optim.lr_scheduler.CosineAnnealingLR',
                                 'learning_rate':2e-5}
                                },
    'path' : {'output' : '../model_checkpoint/image_encoder/',
              'df' : '../data/train.csv',
              'image_dir' : '../data/train_imgs/'}
    }

CFG_BERT = {
    'model' : {'name': 'transformers.',
               'weight': 'cahya/bert-base-indonesian-522M',
               'MAX_LEN': 256,
               'num_classes':11014,
               'scale' : 50,
               'margin' : 0.4},
    'training' : {'batch_size' : 16
                },
    'path' : {'output': '../model_checkpoint/text_encoder/',
              'df' : '../data/train.csv'}
    }
