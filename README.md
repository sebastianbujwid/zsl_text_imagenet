# zsl_text_imagenet

ImageNet-Wiki: matching, processing, etc.
The project contains:
- [ImageNet-Wiki](https://github.com/sebastianbujwid/ImageNet-Wiki_dataset) automatic matching
- Extracting, parsing of Wikipedia articles
- Feature extraction from Wikipedia articles' text

## Conda environment

[conda.yml](./conda.yaml) contains a Conda environment used for the project.
Note that it contains more dependencies than this project requires!

## Issue with downloading ALBERT weights

Due to some changes in newer versions of [transformers](https://github.com/huggingface/transformers) library and models you probably won't be able to download ALBERT weights correctly (most likely will get some logging about that - make sure not to miss them!).
If interested, see more details in the [corresponding Github issue](https://github.com/huggingface/transformers/issues/7889).

To make it work you can try to manually load the cached models we used:
[Older, cached ALBERT models from `transfomers` library](https://kth.box.com/s/rzfsx4eschd88eio8fjh7b53hjdite05)


## Project

The code from this repository was used in our work, see [our project page](https://bujwid.eu/p/zsl-imagenet-wiki).
