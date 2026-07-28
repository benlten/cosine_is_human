### The title is overstated

We acknowledge this and propose "Cosine is Nearly Human: The Reproducibility Ceiling of Perceptual Similarity".

- Zero-shot DINOv1 with plain cosine reaches 74.1% against a split-half floor of 70.1%, untrained.
- The untrained DreamSim ensemble reaches 74.5% and the published checkpoint 80.7%, so the frozen representation supplies 92% of the accuracy.
- 2.0 pp of the remainder is train/test overlap, concentrated on the 20k the checkpoint trained on.
- The residual 4.2 pp holds on triplets where color cannot discriminate. It is also smaller than the noise in the labels it would have to clear: 70.1% reproduction overall, 33 to 50% on the weak-majority triplets that are 46% of the corpus.
- Six of ten frozen non-DINO backbones also reach or pass the floor untrained, one of them BagNet-33, whose 33-pixel receptive field rules out global shape.
- With the representation fixed and only head capacity varying, cosine beats all ten alternatives, and accuracy falls monotonically as capacity rises.
So cosine reaches the ceiling these labels can certify and does not go past it. "Nearly" names that residual without calling it zero and without calling it alignment.

It also leaves the gap open. We do not claim the residual is unreachable, only that supervised triplet training on NIGHTS is not what closes it: the 4.2 pp cannot be separated from voter-sample structure at this label quality, a further 2.0 pp is overlap, and the same training costs measurable invariance on the one axis with a human anchor.


### The paper should include human re-annotation

We agree, and cannot collect it inside the rebuttal window. The protocol is in our reply to Reviewer 95Xn. A new limitations item will say that the bootstrap covers within-population sampling variability only, and that every source it omits pushes the floor down rather than up.


### Larger backbones, non-DINO representations, non-synthetic datasets

We evaluated sixteen backbones zero-shot on NIGHTS-100k with plain cosine and no training, ten of which are not DINO. These results are in Appendices D and F, and we will move them into the main text.


| Backbone, zero-shot cosine | NIGHTS-100k |
|---|---|
| DINOv1 ViT-B/16 | 74.1 ± 0.4 |
| Supervised ViT-B/16 | 72.4 ± 0.2 |
| DINOv3 ViT-B/16 | 72.0 ± 0.4 |
| VOneResNet-50 | 71.8 ± 0.5 |
| CORnet-S / ConvNeXt-L | 71.7 / 71.4 |
| DINOv2 ViT-B/14 | 70.6 ± 0.4 |
| BagNet-33 | 70.3 ± 0.4 |
| BiT-M R101x3 / ResNet-50 | 70.2 / 69.7 |
| CLIP ViT-B/16 / ResNet-152 | 68.0 / 67.1 |
| LPIPS | 65.3 ± 0.2 |

Six of the ten non-DINO backbones reach or pass the 70.1% floor untrained. Limitations item (ii) says "all backbones are ViT-B-class (86M)"; this table contradicts it and will replace it.

The reviewer asks specifically about larger backbones, so we ran a controlled scaling sweep inside one family, covering DINOv2 ViT-S/14 through ViT-g/14. Pretraining, objective, input resolution and folds are all held constant, and only width and depth change.


| DINOv2 variant | Params | NIGHTS-100k | vs. 70.1% floor | Unanimous 20k |
|---|---|---|---|---|
| ViT-S/14 | 21M | 72.19 ± 0.40 | +2.1 | 85.8 |
| ViT-B/14 | 86M | 70.60 ± 0.40 | +0.5 | 83.9 |
| ViT-L/14 | 300M | 68.98 ± 0.22 | −1.1 | 81.8 |
| ViT-g/14 | 1.1B | 68.36 ± 0.24 | −1.7 | 80.5 |

Accuracy falls monotonically as the backbone grows, on both the full corpus and the unanimous subset. Across a 52× range of parameters the smallest model beats the largest by 3.8 pp, and the two largest fall below the reproducibility floor, where they are no longer separable from the label noise. The 21M encoder is the best of the four, and the 1.1B one trails an 86M DINOv1 by 5.7 pp. Per-fold spread runs from 0.22 to 0.40, so none of this is sampling variation. The decline does flatten at the top end, falling 1.6 pp per step up to ViT-L and 0.6 pp from ViT-L to ViT-g, which is what we would expect if accuracy is settling toward the level the labels can support rather than tracking capacity.

We should also credit DreamSim here, because this is not a new observation and the reviewer is right to ask whether it was already known. Fu et al. evaluate CLIP ViT-L/14, MAE ViT-L/16, MAE ViT-H/14 and OpenCLIP ViT-L/14 alongside their ViT-B backbones, and their related work states the trend directly: "While strong computer vision features make for strong perceptual metrics, counterintuitively, they eventually become decorrelated with perceptual similarity." Our contribution is to show that the trend holds in a family they did not test, that it is monotone rather than a plateau, and that on the full 100k it carries the larger models under the reproducibility floor. The revision cites their observation where we report ours.

Scale does not help elsewhere either. ResNet-152 sits below ResNet-50, and BiT-M R101x3 and ConvNeXt-L land within a point of much smaller models. The largest backbone we could afford makes the point hardest: C-RADIOv4-SO400M distills DINOv3, SigLIP2 and SAM3 into one ViT, and with rank-8 LoRA and a cosine head, one epoch on fold 0 reached 58.2% after roughly 28 hours at the 1008×1008 input its RoPE embeddings need. That is 16 pp below frozen DINOv1 and 7 pp below LPIPS.

The furthest we went from a DINO-style representation is a vision-language model answering the 2AFC question directly. Qwen3-VL-4B-Instruct, with randomised A/B order and scored on the logit ratio at the first generated token, reached 63.0% on a 100-triplet sample of fold 0 (95% Wilson interval 53.2 to 71.8) and 55.7% on the 61 weak-majority triplets in it. Both are below LPIPS. That sample places the method but cannot resolve the agreement strata, and we are re-running it at scale. The stratified rows printed in Appendix D are wrong: re-derived from the released fold-0 predictions they are 69.2% on 26 unanimous, 84.6% on 13 strong and 55.7% on 61 weak. The 63.0% total is correct. We are fixing the table and withdrawing the claim that the model reliably calls unanimous triplets, which n = 26 cannot support.

BAPPS is real photographs with five human votes per triplet, and the one place our claim meets an external test. Per category the order inverts: LPIPS beats all three zero-shot DINO backbones on color, deblur and super-resolution (Table R5). Against each benchmark's own reproducibility floor:


| Method | BAPPS (floor 69.0) | NIGHTS-100k (floor 70.1) | Clears both |
|---|---|---|---|
| LPIPS-AlexNet | 72.9 yes | 65.3 no | no |
| DINOv2 ZS | 67.3 no | 70.6 yes | no |
| DINOv3 ZS | 68.4 no | 72.0 yes | no |
| DINOv1 ZS | 71.5 yes | 74.1 yes | yes, untrained |
| DreamSim-Pretrained | 73.6 yes | 80.7 yes | yes, after training |

Two methods clear both floors: DreamSim, after supervised training on a checkpoint that overlaps our test folds, and a frozen DINOv1 with no training at all. Section 4.5 says only DreamSim clears both. That is wrong, and the Appendix H figure already plots DINOv1 above both ceilings. The same paragraph puts the BAPPS gap to LPIPS at 3 to 5 pp; it is 1.4 to 5.6 pp, and DINOv1's 1.4 pp is what clears the floor. Both will be fixed. The error understated our own result.

We will add a scope statement saying that plain cosine on a frozen backbone is at the ceiling for the mid-to-high-level judgments NIGHTS collects, and is not competitive for low-level distortion detection.

One more negative result is worth reporting here. DINOv3 after human-saliency harmonization falls to 64.0 ± 0.2% on NIGHTS-100k, below every untrained backbone in Table R6, so alignment supervised on low-level saliency does not transfer to this regime.
