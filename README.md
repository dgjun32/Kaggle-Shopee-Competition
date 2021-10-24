## 1. Methodology
I applied two-step approach for <b>Product Matching</b> task. 
    
1. Transforming Image and text into representation vectors using fine-tuned encoder model.
2. NearestNeighbors algorithm for matching items, given concatenated *image vector* and *text vector* of each item.

### 1.1. Backbone
* As an Image encoder, pretrained <b>Visual Transformer</b> ```ViT-B/16``` as a backbone.
* As an Text encoder, pretrained <b>```Indonesian BERT```</b> as a backbone.

### 1.2. Learning Metric
<p>I finetuned those backbones using ArcFaceloss, which enhances <b>intra-class compactness</b> and <b>inter-class discrepancy</b> of embedding vectors. <br> Encoder architectures are trained with 60% of the data, with larger learning rate for parameters of cosine head.</p>
<p>After finetuning backbones, I concatenated image embedding and text embedding which are l2-normalized. An then, I found optimal threshold value for matching same product based on concatenated embedding vectors.</p>



## 2. Data
you can download dataframe and images to ```../data/``` directory from the [link](https://www.kaggle.com/c/shopee-product-matching/data).

``` 
cd ../data
gdown "google drive link url"
```

## 3. Training
To train image encoder, command ```python image_main.py```<br>To train text encoder, command ```python text_main.py```

Trained models will be saved as checkpoint file at 

```../model_checkpoint/image_encoder``` and ```../model_checkpoint/text_encoder``` respectively.

## 4. Models


## 5. Evaluation