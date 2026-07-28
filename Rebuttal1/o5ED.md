We thank the reviewer for their helpful comments and address their questions and concerns below.


### Model confidence against the human vote split

We ran this on all 64,420 non-tie and all 35,580 split-vote triplets. Platt scaling fitted on four folds and applied to the fifth maps each method's score to P(left); confidence is 2·|P − 0.5| against the human counterpart 2·|vote fraction − 0.5|.


| Agreement | n | Vote-supported | DreamSim | DINOv1 ZS | LPIPS |
|---|---|---|---|---|---|
| 50/50 | 35,580 | 0.000 | 0.484 | 0.383 | 0.251 |
| 2-1 | 18,454 | 0.333 | 0.492 | 0.386 | 0.250 |
| 4-1 | 6,841 | 0.600 | 0.544 | 0.412 | 0.260 |
| X-0 unanimous | 20,019 | 1.000 | 0.790 | 0.580 | 0.324 |

Confidence rises with agreement but far too slowly, and on the triplets where the votes support zero it still reads 0.484. Our own LoRA-trained heads read 0.80 to 0.95 there, so this is not an artifact of DreamSim's weights. Fig2_rebuttal shows eight 8-0 to 10-0 triplets DreamSim-Ens-Pretrained gets wrong.

We then measured expected calibration error, the average gap between the confidence a method states and how often it is right at that confidence, against two different targets: the binary majority label NIGHTS publishes, and the vote fraction underneath it. The two targets give opposite answers. DreamSim is the best-calibrated method against the published label, at 0.008, and the worst against the vote fraction, at 0.071, while LPIPS scores 0.022 against both. We read that as the trained model having learned the thresholding step that produced the binary label. It did not learn the uncertainty that the threshold discarded. On weak-majority triplets, whose labels reproduce 33 to 50% of the time, 11.8% of its predictions still exceed 0.9 confidence.

The Spearman correlation between P(left) and vote fraction is 0.760 for DreamSim against 0.617 for frozen DINOv1, and the order survives among weak-majority triplets alone (0.482 against 0.373). Fine-tuning does confer sensitivity to human uncertainty the frozen backbone lacks, and the paper now says so.


### Did humans judge the transformed images?

They did not, and no dataset provides them, so we cannot know whether the model's errors there differ from human errors. We cut the two claims that said otherwise.

The eleven manipulations are three rotations, grayscale, HSV value-channel equalization, four blur levels and two radial masks. They characterize the metric under stimulus change, which needs no human data. Rotation, color and blur have no human anchor and move to the appendix. The foreground/background axis stays, because it has one: humans lose about 3 pp under IN-9 background randomization and 1 pp on ObjectNet, while zero-shot DINOv2 loses 2.0 pp and the same backbone after LoRA loses 7.7 pp.

BAPPS is the one corpus with human judgments of distorted images (Appendix H). If DreamSim's low-level sensitivity were unlike human sensitivity it would trail there, but it leads on the two most low-level categories, traditional (82.0 against LPIPS 77.7) and color (67.7 against 66.3). The successor protocol in Section 5 would collect the missing judgments: the same annotators judge the original triplets and the manipulated versions, so the human loss under each manipulation is measured on the same stimuli as the model loss.
