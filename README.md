# Feature Hallucination Using Conditional GAN

### Implementation of the paper - [**An Adversarial Approach to Discriminative Modality Hallucination for Remote Sensing Data**](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CROMOL/Pande_An_Adversarial_Approach_to_Discriminative_Modality_Distillation_for_Remote_Sensing_ICCVW_2019_paper.pdf) - IEEE ICCV - CROMOL 2019 in Keras.


Uses a conditional GAN with novel discriminator of _2C_ classes (corresponsing to _C_ real and fake classes each) to hallucinate missing features for hyperspectral or multimodal remote sensing data using available sensor derived features.

Generator takes latent dimensions as well as conditional input from available modality to generate missing features.
GAN was trained using tips from Soumith Chintala's [GANHacks](https://github.com/soumith/ganhacks).

Deep CNNs have been utilized as feature extractors for application of GAN, and further classification.
Hallucinated features are concatenated with original features and sent to DNN classifier.

Ablation studies done with Indian Pines dataset.

## Poster and Architecture

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








