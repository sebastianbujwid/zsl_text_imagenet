# zsl_text_imagenet

ImageNet-Wiki: matching, processing, etc.
The project contains:
- [ImageNet-Wiki](https://github.com/sebastianbujwid/ImageNet-Wiki_dataset) automatic matching
- Extracting, parsing of Wikipedia articles
- Feature extraction from Wikipedia articles' text

## Download encoded text from Wikipedia articles

Available to download:
[Encoded Wikipedia articles (extracted features)](https://kth-my.sharepoint.com/:f:/g/personal/bujwid_ug_kth_se/EopsN4tGqX1Cuu7QDVSPk2MBLrddGgs0KmwPg5fzgMkNSA?e=FFdzUT). Contains Wikipedia articles corresponding to ImageNet classes encoded with:
- GloVe features
- Word2Vec features
- ALBERT (xxlarge & base)

## Wikipedia dump used

We used `enwiki-20200120` (20 Jan 2020) dump of English Wikipedia, downloaded from [Wikimedia Downloads page](https://dumps.wikimedia.org/).
The original version of the dump that we have used is available for download:
[Original Wikipedia dump we have used](https://kth-my.sharepoint.com/:f:/g/personal/bujwid_ug_kth_se/EtzGiqtVzoxCrwDLxumH7PIBT-EE8nLmj03kE78R8zHlEg?e=7KUP8O).

## Conda environment

[conda.yml](./conda.yaml) contains a Conda environment used for the project.
Note that it contains more dependencies than this project requires!

## Issue with downloading ALBERT weights

Due to some changes in newer versions of [transformers](https://github.com/huggingface/transformers) library and models you probably won't be able to download ALBERT weights correctly (most likely will get some logging about that - make sure not to miss them!).
If interested, see more details in the [corresponding Github issue](https://github.com/huggingface/transformers/issues/7889).

To make it work you can try to manually load the cached models we used:
[Older, cached ALBERT models from `transfomers` library](https://kth-my.sharepoint.com/:f:/g/personal/bujwid_ug_kth_se/EpzO_H4P0t1Biwu8nVdLT7kBr935GCh9U3PM4ncVatbIWg?e=op7lWk)


## Project

The code from this repository was used in our work, see [our project page](https://bujwid.eu/p/zsl-imagenet-wiki).
