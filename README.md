# Shopee Product Matching Algorithm

## 0. What is Product Matching?
Task of product matching is to <b>bind same product together</b>, given data(img, text) of each product.

Current E-commerce company requires product matching algorithm to enhance user experience.(ex.Recommendation system based on product matching)

Final task of this competition is to predict id of same product for each items, thereby maximizing f1-score.

## 1. Methodology
I applied two-step approach for <b>Product Matching</b> task. 
    
1. Transforming Image and text into representation vectors using fine-tuned encoder model.
2. Distance based algorithm for matching items, given concatenated *image vector* and *text vector* of each item.

### 1.1. Backbone
* As an Image encoder, pretrained <b>Visual Transformer</b> ```ViT-B/16``` as a backbone.
* As an Text encoder, pretrained <b>```Indonesian BERT```</b> as a backbone.

### 1.2. Learning Metric
* Finetuned those backbones using ```ArcFaceloss```, which enhances ```intra-class compactness``` and ```inter-class discrepancy``` of embedding vectors.

* Encoder architectures are trained with 60% of the data, with larger learning rate for parameters of cosine head.

* After finetuning backbones, I concatenated image embedding and text embedding which are l2-normalized. An then, I found optimal threshold value for matching same product based on concatenated embedding vectors.</p>


## 2. Data
you can download dataframe and images to ```../data/``` directory from the [link](https://www.kaggle.com/c/shopee-product-matching/data).

``` 
cd ../data
gdown [link for csv file]
gdown [link for image files]
unzip train_imgs.zip
```

## 3. Training
To train image encoder : ```python image_main.py```

To train text encoder : ```python text_main.py```

Trained models will be saved at ```../output``` directory.

## 4. Models
Download image encoder from link to ```../model/image_encoder``` directory
Download text encoder from link to ```../model/text_encoder``` directory

## 5. Result
