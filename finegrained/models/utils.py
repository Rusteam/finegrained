"""Helper functions for training or evaluation.
"""


def validate_train_config(
    cfg: dict,
    data_keys=["dataset", "label_field"],
    model_keys=["backbone"],
    trainer_keys=["epochs"],
):
    for key in ["data", "model", "trainer"]:
        assert key in cfg, f"config file has to contain {key=}"
    for key in data_keys:
        assert key in cfg["data"], f"data has to contain {key=}"
    for key in model_keys:
        assert key in cfg["model"], f"model has to contain {key=}"
    for key in trainer_keys:
        assert key in cfg["trainer"], f"trainer has to contain {key=}"
