CFG_VIT = {
    'model' : {'name': 'transformers.ViTForClassification',
               'weight': 'google/vit-base-patch16-224',
               'img_size' : 224,
               'num_classes' : 11014,
               'scale' : 50,
               'margin' : 0.4},
    'training' : {'batch_size':16,
                  'epochs' : 10,
                  'lr_schedule':{'name':'tf.keras.optimizers.schedules.CosineDecayRestarts',
                                 'learning_rate':2e-5,
                                 'first_decay_steps':500, 
                                 't_mul':2.0, 
                                 'm_mul':1.0,
                                 'alpha':0.0}
                },
    'path' : {'output' : '../model_checkpoint/image_encoder/',
              'df' : '../data/',
              'image_dir' : '../data/train_imgs/'}
    }

config_BERT = {
    'model' : {'name': 'cahya/bert-base-indonesian-522M',
               'MAX_LEN': 256,
               'num_classes':11014,
               'scale' : 50,
               'margin' : 0.4},
    'training' : {'batch_size' : 16,
            }
    
}
