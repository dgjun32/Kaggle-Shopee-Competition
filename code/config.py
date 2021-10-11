config_arcface = {
    'model' : {'name': 'tf.keras.applications.EFFicientNetB3',
               'img_size' : 512,
               'num_classes' : 11014,
               'scale' : 50,
               'margin' : 0.4},
    'training' : {'batch_size':16,
                  'epochs' : 10,
                  'lr_schedule':{'name':'tf.keras.optimizers.schedules.CosineDecayRestarts'
                                 'learning_rate':2e-5,
                                 'first_decay_steps':500, 
                                 't_mul':2.0, 
                                 'm_mul':1.0,
                                 'alpha':0.0}
                }
    'checkpoint' : {'path' : '../model_checkpoint/image_encoder/image_model.h5'}
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
