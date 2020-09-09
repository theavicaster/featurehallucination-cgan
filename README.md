# Feature Hallucination Using Conditional GAN

### Implementation of the paper - [**An Adversarial Approach to Discriminative Modality Hallucination for Remote Sensing Data**](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CROMOL/Pande_An_Adversarial_Approach_to_Discriminative_Modality_Distillation_for_Remote_Sensing_ICCVW_2019_paper.pdf) - IEEE ICCV - CROMOL 2019 in TensorFLow


Uses a conditional GAN with novel discriminator of _2C_ classes (corresponsing to _C_ real and fake classes each) to hallucinate missing features for hyperspectral or multimodal remote sensing data using available sensor derived features.

Generator takes latent dimensions as well as conditional input from available modality to generate missing features.
GAN was trained using tips from Soumith Chintala's [GANHacks](https://github.com/soumith/ganhacks).

Deep CNNs have been utilized as feature extractors for application of GAN, and further classification.
Hallucinated features are concatenated with original features and sent to DNN classifier.

Ablation studies done with Indian Pines dataset.

## Poster and Architecture

![Poster Presented](https://github.com/theavicaster/featurehallucination-cgan/blob/master/poster_iccv.png)

## t-SNE Plots

Without sensor abnormality           |  With missing bands due to failures
:-------------------------:|:-------------------------:
![](https://github.com/theavicaster/featurehallucination-cgan/blob/master/tsneorig.png) |  ![](https://github.com/theavicaster/featurehallucination-cgan/blob/master/tsnehall.png)


## Datasets

* Indian Pines Corrected
    * http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
* Pavia University Scene
    * http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
* Houston Fusion
    * http://hyperspectral.ee.uh.edu/?page_id=1075
* Multimodal Dataset
    * Uses Multispectral and complementary Panchromatic data.
    * Dataset is not publicly available.
    
## Results

### Train Test Split Experiments
#### (Temperature 2, 50-50 Band Split)


|  Split | 75-25 | 50-50 | 25-75 | 10-90 |
| ------------- | ------------- | ----- | ---| ------|
| Teacher Network | 97.54194303550526% | 90.71219512195122% | 70.28749837697956% | 69.51761517809012% |
| Student Network  | 93.79633242294186% | 92.13658536701668% | 80.57759854532085% | 71.92411924248465% |


### Temperature Experiments
#### (25-75 Train Test Split, 50-50 Band Split)


|  Temperature | 1 | 2 | 3 | 5 | 10 |
| ------------- | ------------- | ----- | ---| ------| --- |
| Teacher Network | 69.4289059482037%| 85.625081307652%| 72.19981788053645% | 81.63132561622491% | 57.47365682630959%|
| Student Network  | 77.31234552461086%|  95.4728762846364% | 79.84909587848074%| 84.75347990733494%| 64.39443216129073% |


### Band Split Experiments (Hallucinated-Real)
#### (Temperature 2, 25-75 Train Test Split)

|  Split | 25-75 | 50-50 | 75-25 | 90-10 |
| ------------- | ------------- | ----- | ---| ------|
| Teacher Network | 55.60036425191497%| 85.625081307652% | 62.29998699334999%| 64.70664758799801%|
| Student Network  | 56.70612723088876%| 95.4728762846364%| 86.09340444906985% | 81.43619097332131% |



## Usage

Run the files in following order, modifying the directory of dataset -

```sh
python twostreamdistillation.py
python train.py
```
The file testmodels.py is used to evaluate the optimal model for generator.

```sh
python testmodels.py
python test.py
```
## Citation

If this paper and/or this code is useful for your research, please consider citing us -

```
@inproceedings{pande2019adversarial,
  title={An Adversarial Approach to Discriminative Modality Distillation for Remote Sensing Image Classification},
  author={Pande, Shivam and Banerjee, Avinandan and Kumar, Saurabh and Banerjee, Biplab and Chaudhuri, Subhasis},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```








