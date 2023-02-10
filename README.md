# Ischemic_Stroke_Prediction
Predicting, whether patient has Ischemic Stroke from brain CT images.

If your machine doesn't have gpu, use env files from folder 'poetry_no_gpu'.

- Run this command to start train new model.
```bash
poetry run python scenarios/training_scenario.py dataset_folder
```
To see all training options run this.
```bash
poetry run python scenarios/training_scenario.py --help
```

- To finetune already trained model run
```bash
poetry run python scenarios/futher_training.py dataset_folder checkpoint_file.ckpt
```
To see all fine-tuning options run this.
```bash
poetry run python scenarios/futher_training.py --help
```

- To test your model open 'scenarios/testing_scenario.py' in IDE and change `test_dir` and `checkpoint_path` params.