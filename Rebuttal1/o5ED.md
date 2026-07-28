### Model confidence against the human vote split

At the reviewer's suggestion we have run this experiment, on all 64,420 non-tie and all 35,580 split-vote triplets.

Each method's score becomes P(left) by Platt scaling fitted on four folds and applied to the fifth, so every probability is out-of-sample. Model confidence is 2·|P − 0.5|. Its human counterpart is 2·|vote fraction − 0.5|: 1.0 on a unanimous triplet, 0 on a 50/50 one. The split-vote triplets never enter the fit.


| Agreement | n | Vote-supported | DreamSim | DINOv1 ZS | DINOv2 ZS | LPIPS |
|---|---|---|---|---|---|---|
| 50/50 | 35,580 | 0.000 | 0.484 | 0.383 | 0.321 | 0.251 |
| 2-1 | 18,454 | 0.333 | 0.492 | 0.386 | 0.325 | 0.250 |
| 3-1 | 11,181 | 0.500 | 0.521 | 0.409 | 0.340 | 0.256 |
| 4-1 | 6,841 | 0.600 | 0.544 | 0.412 | 0.346 | 0.260 |
| X-1 strong | 7,922 | 0.697 | 0.586 | 0.438 | 0.360 | 0.271 |
| X-0 unanimous | 20,019 | 1.000 | 0.790 | 0.580 | 0.467 | 0.324 |

Confidence does rise with agreement, and too slowly. DreamSim moves 0.492 to 0.790 while the votes move 0.333 to 1.000. Repeating the sweep cumulatively from the published 20k out to the full 100k, vote-supported confidence falls 1.000 to 0.414 and DreamSim's falls 0.790 to 0.563. The curves cross near 70,000 triplets, beyond which the model is more certain than the votes it was fitted to.

Separating the two calibration targets reverses the ranking.


| Method | ECE vs. majority label | ECE vs. vote fraction | Conf. on 50/50 |
|---|---|---|---|
| LPIPS | 0.022 | 0.022 | 0.251 |
| DINOv2 ZS | 0.024 | 0.035 | 0.321 |
| DINOv3 ZS | 0.015 | 0.042 | 0.353 |
| DINOv1 ZS | 0.017 | 0.047 | 0.383 |
| DreamSim-Pretrained | 0.008 | 0.071 | 0.484 |

DreamSim is the best-calibrated method against the label NIGHTS publishes and the worst against the vote distribution underneath it. We read that as the trained model learning the thresholding step that produced the binary label, not the uncertainty the threshold discarded. On triplets where the votes support a confidence of zero it reads 0.484, and 11.8% of its weak-majority predictions exceed 0.9 confidence, on labels that reproduce 33 to 50% of the time.

Our own LoRA-trained heads are worse than any of them and serve as a minimal control against the DINO models.


| Trained head, native sigmoid | Mean conf. on 50/50 | Fraction > 0.9 |
|---|---|---|
| DINOv2 + Tversky/GCM | 0.946 | 88.5% |
| DINOv1 + Tversky/GCM | 0.850 | 68.8% |
| DINOv3 + Tversky/GCM | 0.802 | 59.7% |
| TNN head | 0.536 | 14.8% |

Fig2_rebuttal in the anonymous repository shows eight 8-0 to 10-0 triplets that DreamSim-Ens-Pretrained gets wrong, the errors label noise does not explain.

These were trained on our own 100k splits with no train/test overlap, so this is not an artifact of DreamSim's published weights. The submission already used this subset once, for the positional-bias audit in Appendix F, where every embedding method sits between 49.5% and 51.1% left-preference. The models are not defaulting to a side.

One result runs against us. The Spearman correlation between calibrated P(left) and the vote fraction is 0.760 for DreamSim, 0.617 for zero-shot DINOv1, 0.406 for LPIPS. We expected this to reflect difficulty rather than uncertainty, since easy triplets have both large margins and high agreement, but restricting to weak-majority triplets leaves the order intact (0.482, 0.373, 0.218). Fine-tuning buys sensitivity to human uncertainty that the frozen backbone lacks. We have added this finding to the paper. We note that for the remaining 80% the disagreement is high, and that although we cannot prove it without human follow-up, on the in-agreement 20% a different set of judges would plausibly change this result.


### Did humans judge the transformed images?

We did not use human judges for these stimuli. The reason is that the variability we found in the existing labels is what prompted this analysis in the first place; the motivation is to build a more tightly controlled generator for human collection. NIGHTS has no human votes on rotated, blurred, grayscale or masked triplets, and we have collected none.

The eleven manipulations were built to characterize the metric, not to compare it against people: whether something that scores well on clean triplets still behaves like a similarity metric once the stimulus is altered. That is a model property and needs no human data. Our manipulation set follows four of the five axes of DreamSim's Figure 6. We are keeping the ablation.

Because NIGHTS reports no per-rater judgments, we cannot calibrate the variance either. No other perceptual-similarity dataset provides human judgments on these axes.


| Claim | Evidence | Status |
|---|---|---|
| Fine-tuning reduces invariance on the axes it touches | model-only | kept in main text |
| LoRA moves the model out of a human-compatible regime | IN-9, ObjectNet | kept, rewritten as a cross-dataset comparison |
| DINOv3 is the most invariant zero-shot backbone | model-only, no human anchor | moved to the appendix |
| DreamSim has the best clean score and the worst drops on rotation, color, blur | model-only, no human anchor | moved to the appendix |
| The 4.6-pp gain buys background shortcuts humans do not use | none | cut |
| Model errors on transformed stimuli differ from human errors | none | cut |

Following from that, the per-axis results for rotation, color and blur move to the appendix. Those axes have no human anchor of any kind, so they cannot carry a claim about human perception, and leaving them in the main text invites exactly the reading the reviewer objects to. What stays in the main text is the aggregate invariance-versus-accuracy trade-off, which is a statement about models, and the foreground/background axis, which has an anchor.

One axis of the eleven has an independent human anchor. Humans lose about 3 pp under IN-9 background randomisation and about 1 pp on ObjectNet. Zero-shot DINOv2 loses 2.0 pp under foreground-only masking, inside that range; after LoRA it loses 7.7 pp, outside it. Those anchors are a different dataset and a categorisation task rather than a similarity judgment, and the revision says so where it uses them. The other ten axes support no human comparison.

BAPPS is the one perceptual corpus with human judgments of distorted images, five votes per triplet. If DreamSim's low-level sensitivity were unlike human sensitivity it should trail there. It leads.


| Method | traditional | cnn | color | deblur | frameinterp | superres |
|---|---|---|---|---|---|---|
| LPIPS | 77.7 | 86.7 | 66.3 | 63.8 | 66.9 | 76.2 |
| DINOv1 | 80.0 | 85.2 | 62.0 | 62.4 | 66.1 | 73.2 |
| DINOv2 | 75.4 | 81.0 | 58.1 | 59.1 | 66.6 | 63.7 |
| DINOv3 | 77.2 | 83.2 | 59.8 | 60.2 | 64.4 | 65.8 |
| DreamSim | 82.0 | 86.8 | 67.7 | 62.6 | 67.5 | 75.0 |

We will narrow the claim so that it says only what the evidence supports: fine-tuning reduces invariance, and only the foreground/background axis has a human anchor suggesting the loss moves away from human behavior. The re-annotation study in our reply to Reviewer 95Xn would collect human judgments on these altered images.
