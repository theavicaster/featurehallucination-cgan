# Feature Hallucination Using Conditional GAN

>Implementation of the paper - An Adversarial Approach to Discriminative Modality Hallucination for Remote Sensing Data - ICCVw CROMOL 2019 in Keras.

>http://openaccess.thecvf.com/content_ICCVW_2019/papers/CROMOL/Pande_An_Adversarial_Approach_to_Discriminative_Modality_Distillation_for_Remote_Sensing_ICCVW_2019_paper.pdf

Uses a conditional GAN with discriminator of 2C classes for real and fake to hallucinate missing features for hyperspectral or multimodal data using available features.

Generator takes latent dimensions as well as input from available modality to generate features.
GAN was trained using tips from https://github.com/soumith/ganhacks

Hallucinated features are concatenated with original and sent through to classifier.

Ablation studies done with Indian Pines dataset.

## Poster

![Poster Presented](https://github.com/theavicaster/featurehallucination-cgan/blob/master/poster_iccv.png)

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









