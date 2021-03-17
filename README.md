# zsl_text_imagenet

ImageNet-Wiki: matching, processing, etc.
The project contains:
- [ImageNet-Wiki](https://github.com/sebastianbujwid/ImageNet-Wiki_dataset) automatic matching
- Extracting, parsing of Wikipedia articles
- Feature extraction from Wikipedia articles' text

## Wikipedia dump used

We used `enwiki-20200120` (20 Jan 2020) dump of English Wikipedia, downloaded from [Wikimedia Downloads page](https://dumps.wikimedia.org/).
The original version of the dump that we have used is available for download:
[Original Wikipedia dump we have used](https://kth.box.com/s/0omtdfafycz7cb6kh3lxx16lxny4v9ni).

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
