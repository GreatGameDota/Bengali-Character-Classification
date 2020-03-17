# Bengali Character Classification

My 34th place solution and writeup for the [Bengali Character Classification Compeittion](https://www.kaggle.com/c/bengaliai-cv19) hosted on Kaggle by BengaliAi

<img src='https://github.com/GreatGameDota/Bengali-Character-Classification/blob/master/assets/comp.png?raw=true' alt='comp' title='comp' height=50 width=50>

## Initial Thoughts

<i>DISCLAIMER: This repo does not contain the code for the 34th place solution. Reason being that solution was a pipeline I made a month before the competition ended. So I don't have any of the code for it or the trained weights. I will however summarize what I can remember the solution being as well as talk about the final solution that got 203rd on the private leaderboard.</i>

Overall this competition was extremely enjoyable. I learned how Pytorch[[4]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts) works, learned how to augment data, and even teamed up for the first time! I am very glad to get my first ever medal and have a solution that scored so high on the leaderboard. At the end of this competition I learned a whole lot about public vs private leaderboard scores and learned a lot from reading the amazing summaries and solution by the top teams! Now on to my solution!

## Overview

34th place solution:  
This solution used a pretrained seresnext50 model with a complicated head. Trained on 128x128 preprocessed images[[1]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts) with Affine augmentations[[2]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts) and mixup/cutmix[[3]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts). Adam optimizer with ReduceLROnPlateau.

203rd place solution:  
This solution was a blend of many different models trained by my teammates and I. It included 4 of my seresnext50 models, some efficientnet b4 models and some others. I won't go into detail about my teammates models.

## Model

34th place model:  
All my models are very simple: pretrained seresnext50 with 3 heads for grapheme root, vowel, and consonant. For this solution view the below figure:

![](https://github.com/GreatGameDota/Bengali-Character-Classification/blob/master/assets/model.png)

203rd place model:  
This solution used 4 seresnext50 models trained via the same pipeline on different folds. They have simple heads: Seresnext50 -> AdapdiveAvgPool2d -> Flatten -> Dropout -> Linear.

All seresnext50 pretrained weights were loaded from Pytorchcv[[4]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts). All models loss functions depended on whether mixup or cutmix was applied to the batch (see utils/MixupCutmix.py).

## Input and Augmentation

Input for this competition was pretty simple. I only used the given data converted to images. I used no external data (more on that later). The tough part of this competition was augmentation.

34th place input:  
This solution used the 128px by 128px preprocessed images made by @lafoss[[1]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts). For augmentation I used Affine augmentations by @corochann[[2]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts) as well as the mixup and cutmix implementation by @MachineLP[[3]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts).

203rd place input:  
This solution used just 137px by 236px original images. Albumentations[[4]](https://github.com/GreatGameDota/Bengali-Character-Classification#final-thoughts) for augmentations which include SSR, Cutout, IAAAffine, and IAAPerspective. This solution also used mixup and cutmix.

## Training

I experimented a lot with training over the coarse of this competition. Different optimizers, intital learning rates, batch size, schedulers, mixup/cutmix.

34th place training:  
For this solution I used AdamW with initial learning rate of 0.001, ReduceLROnPlateau as scheduler, 50% of batches use mixup, 50% use cutmix, and trained for around 50 epochs. This solution was trained on Google Colab.

203rd place training:  
For this solution I used Adam with initial learning rate of 0.00016, ReduceLROnPlateau as scheduler, same 50/50 split of mixup/cutmix, and trained for 100 epochs. This solution was trained on my own RTX 2080 I bought midway through this competition.

## Final Submission

For final submission my team choose a blend of many different models. This blend include 4 of my seresnext50 models.

```c
Public LB: .9864
Private LB: .9311
```

However the 34th place solution was not chosen which was a single seresnext50 I trained a month before the competition ended.

```c
Public LB: .9661
Private LB: .9407
```

## Final Thoughts

This competition was a blast! I learned a lot and cannot wait to bring this knowledge to my next competition!  
MANY thanks to my teammates who helped me in the final week:
[Corey Levinson](https://www.kaggle.com/returnofsputnik), [Parker Wang](https://www.kaggle.com/cswwp347724), [Rob Mulla](https://www.kaggle.com/robikscube), and [Balaji Selvaraj](https://www.kaggle.com/dhakshiin1601)!  
I am very excited to get my first competition medal and can't wait to obtain more!  
Now on to the next competition and happy modeling!

My previous competition: [6DoF Car Detection Competition](https://github.com/GreatGameDota/6DoF-Car-Detection-from-RGB-Images)

My next competition: [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)

---

- [1] Preprocessed 128x128 images kernal by lafoss https://www.kaggle.com/iafoss/image-preprocessing-128x128

- [2] Affine Augmentation implementation and kernal by corochann https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch

- [3] Mixup/Cutmix with OHEM loss implementation by MachineLP https://www.kaggle.com/c/bengaliai-cv19/discussion/128637

- [4] My thanks to the contributers and makers of Pytorch, PytorchCV, and Albumentations