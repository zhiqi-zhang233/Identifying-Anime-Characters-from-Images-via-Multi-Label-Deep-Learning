
```text
project_root/
  data/
    images/           # 6000 张图片
    labels.csv        # 每张图的多标签
  src/
    dataset.py        # 全组共享
    transforms.py     # 全组共享
    evaluate.py       # 全组共享
    train_cnn.py
    train_transformer.py
    train_loss.py
  results/
    cnn/
    transformer/
    loss/
```
