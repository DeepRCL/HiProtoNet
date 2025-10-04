
# HiProtoNet

Official code for:
**HiProtoNet: Hyperbolic Hierarchy-aware Part Prototypes for Aortic Stenosis Severity Classification**
MICCAI 2025 ASMUS Workshop

## Quick Start

1. **Clone the repository**
	```powershell
	git clone https://github.com/DeepRCL/HiProtoNet.git
	cd HiProtoNet
	```

2. **Install dependencies**
	```powershell
	pip install --upgrade pip
	pip install torch torchvision torchaudio
	pip install moviepy==1.0.3 pandas wandb tqdm seaborn torch-summary opencv-python jupyter jupyterlab tensorboard tensorboardX imageio array2gif scikit-image scikit-learn torchmetrics termplotlib
	pip install --upgrade plotly einops transformers timm
	pip install git+https://github.com/geoopt/geoopt.git
	pip install --upgrade wandb
	pip install -e .
	```

3. **Prepare your data**
	- Place your data in the `data/` folder.
	- For custom datasets, refer to `src/data/` for dataloader examples.

## Training & Testing

Run training or evaluation from the project root:

```powershell
python main.py --config_path="src/configs/<config-name>.yml" --run_name="<your_run_name>" --save_dir="logs/<your_run_name>"
```

**Common options:**
- `--eval_only=True` : Evaluate a trained model
- `--eval_data_type="val"` or `"test"` : Validation or test set
- `--push_only=True` : Project prototypes to nearest features

**Example:**
```powershell
python main.py --config_path="src/configs/Hyperbolic_XProtoNet.yml" --run_name="HiProtoNet_test" --save_dir="logs/HiProtoNet/test_run"
```


## Acknowledgements

Parts of the codebase related to Lorentzian geometry (e.g., `src/utils/lorentz.py`) are adapted from the MERU project:
- **MERU: Hyperbolic Image-Text Representations** (ICML 2023)
- Code: https://github.com/facebookresearch/meru

## Developer Notes

HiProtoNet builds on ProtoASNet (MICCAI 2023). Most code is inherited; new features are in files with `Hyperbolic` or `Lorentz` in their names:

**Key scripts for hyperbolic/Lorentz models:**
- `src/agents/Hyperbolic_XProtoNet.py`, `Hyperbolic_XProtoNet_e2e.py`, `Hyperbolic_Video_XProtonet_e2e.py`
- `src/models/Hyper_XProtoNet.py`, `Hyper_Video_XProtoNet.py`
- `src/utils/lorentz.py`


Other files follow ProtoASNet conventions. For more details on ProtoASNet, see the DeepRCL fork: https://github.com/DeepRCL/ProtoASNet

## Citation
If you use this code, please cite our paper:

```bibtex
@InProceedings{10.1007/978-3-032-06329-8_19,
	author    = {Vaseli, Hooman and Wu, Victoria and Kim, Diane and Tsang, Michael Y. and Gu, Ang Nan and Luong, Christina and Abolmaesumi, Purang and Tsang, Teresa S. M.},
	editor    = {Ni, Dong and Noble, Alison and Huang, Ruobing and Xue, Wufeng},
	title     = {HiProtoNet: Hyperbolic Hierarchy-Aware Part Prototypes for Aortic Stenosis Severity Classification},
	booktitle = {Simplifying Medical Ultrasound},
	year      = {2026},
	publisher = {Springer Nature Switzerland},
	address   = {Cham},
	pages     = {197--207},
	isbn      = {978-3-032-06329-8}
}
```
