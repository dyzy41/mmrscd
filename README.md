The Pytorch implementation for:
“EfficientCD: A New Strategy For Change Detection Based With Bi-temporal Layers Exchanged[]([[2407.15999/] EfficientCD: A New Strategy For Change Detection Based With Bi-temporal Layers Exchanged (arxiv.org)](https://arxiv.org/abs/2407.15999)),
[Sijun Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong,+S), [Yuwei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+Y), [Geng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+G), [Xiaoliang Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng,+X)::yum::yum:



![image-20240724222528684](./docs/en/image-20240724222528684.png)

### Requirement  
 [env.yaml](env.yaml) 


## Revised parameters 
check the   [configs](configs) 

## Training, Test and Visualization Process   

```bash
bash tools/train.sh
```


![image-20240724223103482](./docs/en/image-20240724223103482.png)

![image-20240724223124411](./docs/en/image-20240724223124411.png)

![image-20240724223137020](./docs/en/image-20240724223137020.png)

![image-20240724223150191](./docs/en/image-20240724223150191.png)

## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{10608163,
  author={Dong, Sijun and Zhu, Yuwei and Chen, Geng and Meng, Xiaoliang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={EfficientCD: A New Strategy For Change Detection Based With Bi-temporal Layers Exchanged}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Remote sensing;Task analysis;Computational modeling;Transformers;Biological system modeling;Land surface;Change detection;feature interaction;Euclidean distance},
  doi={10.1109/TGRS.2024.3433014}}
```
## Acknowledgments

 Our code is inspired and revised by [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation),  [timm](https://github.com/huggingface/pytorch-image-models). Thanks  for their great work!!  



## Reference  

