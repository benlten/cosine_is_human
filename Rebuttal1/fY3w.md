We thank the reviewer for their helpful comments and address their questions and concerns below.


### The title is overstated

We accept the point and propose "Cosine is Nearly Human: The Reproducibility Ceiling of Perceptual Similarity". Zero-shot DINOv1 reaches 74.1% untrained against a 70.1% floor, and the untrained DreamSim ensemble reaches 74.5% against its checkpoint's 80.7%, so the frozen representation supplies 92% of the accuracy. Of the remainder, 2.0 pp is train/test overlap on the 20k the checkpoint trained on. The residual 4.2 pp survives a color control but stays under the label noise it would have to clear. We use "nearly" because that residual exists and we are not claiming it is zero. We are also not claiming it is unreachable, only that supervised triplet training on NIGHTS does not reach it.


### The paper should include human re-annotation

We did not collect more annotation because we already know what it would show. The task collapses too many axes of similarity into a single bit, so a new group answering the same question would disagree in the same way and confirm the result we report. What NIGHTS needs is a collection that separates those axes and records per-rater votes, and Section 5 specifies that protocol: 3,000 triplets stratified by agreement pattern, 30 annotators each, no early stopping, and per-rater votes in the release.


### Larger backbones, non-DINO representations, non-synthetic datasets

We evaluated sixteen backbones zero-shot with plain cosine and no training. Ten are not DINO, spanning 65.3% (LPIPS) to 72.4% (supervised ViT-B/16), and six of them reach or pass the 70.1% floor. Limitations item (ii) claims all backbones are ViT-B-class, which this contradicts, and we are replacing it with the full table. For scale we swept one family, holding pretraining, objective, resolution and folds constant.


| DINOv2 | Params | NIGHTS-100k | vs. floor | Unanimous 20k |
|---|---|---|---|---|
| ViT-S/14 | 21M | 72.19 ± 0.40 | +2.1 | 85.8 |
| ViT-B/14 | 86M | 70.60 ± 0.40 | +0.5 | 83.9 |
| ViT-L/14 | 300M | 68.98 ± 0.22 | −1.1 | 81.8 |
| ViT-g/14 | 1.1B | 68.36 ± 0.24 | −1.7 | 80.5 |

Accuracy falls monotonically across a 52× parameter range. The 21M model beats the 1.1B by 3.8 pp, and the two largest fall below the floor. C-RADIOv4-SO400M, distilling DINOv3, SigLIP2 and SAM3, reached 58.2% after one epoch on fold 0, which is 16 pp below frozen DINOv1. Qwen3-VL-4B answering the 2AFC question directly reached 63.0% on a 100-triplet sample (Wilson 53.2 to 71.8), also below LPIPS; its stratified rows in Appendix D are wrong and we are correcting them.

DreamSim reported this trend first, and Fu et al. evaluate CLIP ViT-L/14, MAE ViT-L/16 and MAE ViT-H/14, and state that strong features "eventually become decorrelated with perceptual similarity". We add that it holds in a family they did not test, that it is monotone across all four sizes, and it places the larger models below the floor.

On non-synthetic data, BAPPS inverts the per-category ranking against our prediction: LPIPS beats every zero-shot DINO backbone on color, deblur and super-resolution, so plain cosine is not competitive for low-level distortion detection, which the abstract will now say. Measured against each benchmark's own floor, two methods clear both, DreamSim at 73.6 and 80.7 and a frozen DINOv1 at 71.5 and 74.1. Section 4.5 says only DreamSim does, which is wrong, and we are correcting it, along with the BAPPS gap, which is 1.4 to 5.6 pp and not 3 to 5. One further negative result: DINOv3 after human-saliency harmonization falls to 64.0%, below every untrained backbone we tested.
