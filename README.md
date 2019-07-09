# featurehallucination-cgan

Uses a conditional GAN with discriminator of 2C classes for real and fake to hallucinate missing features for hyperspectral or multimodal data using available features.

Hallucinated features are concatenated with original and sent through to classifier.

Results - 

Indian Pines Dataset

Train Test Split Experiments(Temperature 2, 50-50 Band Split)

75-25
Teacher - 97.54194303550526%
Student - 93.79633242294186%

50-50
Teacher - 90.71219512195122%
Student - 92.13658536701668%

25-75
Teacher - 70.28749837697956%
Student - 80.57759854532085%

10-90
Teacher - 69.51761517809012%
Student - 71.92411924248465%


Temperature Experiments(25-75 Train Test Split, 50-50 Band Split)

T1
Teacher - 69.4289059482037%
Student - 77.31234552461086%

T2
Teacher - 85.625081307652%
Student - 95.4728762846364%

T3
Teacher - 72.19981788053645%
Student - 79.84909587848074%

T5
Teacher - 81.63132561622491%
Student - 84.75347990733494%

T10
Teacher - 57.47365682630959%
Student - 64.39443216129073%


Band Split Experiments(Hallucinated-Real)(Temperature 2, 25-75 Train Test Split)

25-75
Teacher - 55.60036425191497%
Student - 56.7061272cd3088876%

50-50
Teacher - 85.625081307652%
Student - 95.4728762846364%

75-25
Teacher - 62.29998699334999%
Student - 86.09340444906985%

90-10
Teacher - 64.70664758799801%
Student - 81.43619097332131%


Pavia University Dataset (Temperature 2, 25-75 Train Test Split, 50-53 Band Split)

Teacher - 98.17966460795201%
Student - 79.40589738720024%


Houston Fusion Dataset (Temperature 2, 25-75 Train Test Split, 100-44 Band Split)

Teacher - 98.17701961759527%
Student - 97.9696530922002%


Multimodal Dataset (Temperature 2, 25-75 Train Test Split)

Multispectral Hallucinated
Teacher - 95.46666666666667%
Student - 82.75%

Panchromatic Hallucinated
Teacher - 95.46666666666667%
Student - 86.40333333333333%










