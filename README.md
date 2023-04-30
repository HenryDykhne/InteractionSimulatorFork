## Setup
1. Setup InteractionSimulator Requirements
```
pip install -r requirements.txt
pip install -e .

```
2. Follow instructions to install [ai-traineree](https://github.com/laszukdawid/ai-traineree) via `Git repository clone`.
3. Install version 1.5.0 of petting zoo
4. Download the [INTERACTION dataset](https://interaction-dataset.com/)
5. Run `format_dataset.py` to format the interaction dataset
```
python3 format_dataset.py
```
## Running
```
python3 test_simulator.py
```

