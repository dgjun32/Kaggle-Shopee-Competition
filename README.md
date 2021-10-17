## 1. Methodology
I applied two-step approach for <b>Product Matching</b> task. 
    
    1. Image, Text encoder for transforming Image and text into representation vector.
    
    2. ML algorithm for grouping items, given image vector and text vector of each item.

### 1.1. Backbone
As an Image encoder, I finetuned pretrained <b>Visual Transformer</b> ```ViT-B/16``` by adding only one linear layer on top of the backbone. 

As an Text encoder, I finetuned pretrained <b>Indonesian BERT</b>. 

### 1.2. Learning Metric
I finetuned those backbones using ArcFace loss, which can enhance <b>intra-class compactness</b> and <b>inter-class discrepancy</b> of embedding vectors. 

## 2. Data
you can download dataframe and images to ```../data/``` directory.

<pre><code> 
cd ../data
gdown "sharing link url"
</code></pre>

## 3. Training
To train image encoder, command ```python image_main.py```

To train text encoder, command ```python text_main.py```

Trained models will be saved as checkpoint file at ```../model_checkpoint/image_encoder``` and ```../model_checkpoint/text_encoder``` respectively.

## 4. Models


## 5. Evaluation