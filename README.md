# Triple-GNNs
This is code for Triple-GNNs.

## RUN
`sh run_zh.sh` for ZH dataset

`sh run_en.sh` for EN dataset
## Resource Requirement
### Environment
- Pytorch 1.12.1+cu113
- Python 3.8.10
- spacy 3.4.1

### Dependency Parser
1. `en_core_web_sm-3.3.0` for EN dataset. **[DOWNLOAD](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.3.0)**
2. `zn_core_web_sm-3.3.0` for ZH dataset. **[DOWNLOAD](https://github.com/explosion/spacy-models/releases/tag/ZH_core_web_sm-3.3.0)**

### Device
1. 12GB GPU for ZH dataset.
2. 16GB GPU for EN dataset.

## Thanks
This code is based on [DiaASQ](https://github.com/unikcc/DiaASQ) and [DualGAT](https://github.com/something678/TodKat). Thanks for their outstanding work.

### Cite
If you use our code or dataset in your research, please cite:

