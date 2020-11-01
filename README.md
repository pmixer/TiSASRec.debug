For debugging the well-know issue of performance over-estimation addressed in https://www.kdd.org/kdd2020/accepted-papers/view/on-sampled-metrics-for-item-recommendation, SASRec&TiSASRec are still good with high speed and accurate prediction, we are just trying to make it better without negative sampling based evaluation.

---

# TiSASRec: Time Interval Aware Self-Attention for Sequential Recommendation

This is our TensorFlow implementation for the paper:

Jiacheng Li, Yujie Wang, [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/) (2020). *[Time Interval Aware Self-Attention for Sequential Recommendation.](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf)* WSDM'20

We refer to the repo [SASRec](https://github.com/kang205/SASRec).

Please cite our paper if you use the code or datasets.

The code is tested under a Linux desktop (w/ GTX 1080 Ti GPU) with TensorFlow.

For Pytorch version of TiSASRec, please refer to [repo](https://github.com/pmixer/TiSASRec.pytorch).

## Datasets

This repo includes ml-1m dataset as an example.

For Amazon dataset, you could download Amazon review data from *[here.](http://jmcauley.ucsd.edu/data/amazon/index.html)*.

## Model Training

To train our model on `ml-1m` (with default hyper-parameters): 

```
python main.py --dataset=ml-1m --train_dir=default 
```

## Misc

The implemention of self attention is modified based on *[this](https://github.com/Kyubyong/transformer)*.

## Contact

If you have any questions, please send me an email (j9li@eng.ucsd.edu).

