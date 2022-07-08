def check_triton_onnx(model_dir):
    assert model_dir.exists()
    assert (model_dir / "config.pbtxt").exists()
    assert (model_dir / "1" / "model.onnx").exists()


def check_triton_python(export_dir):
    assert (export_dir / "tokenizer" / "1" / "model.py").exists()
    assert (export_dir / "tokenizer" / "config.pbtxt").exists()


def check_triton_ensemble(model_dir):
    assert (model_dir / "ensemble" / "1").exists()
    assert (model_dir / "ensemble" / "config.pbtxt").exists()


def check_triton_labels(model_dir):
    assert (model_dir / "labels.txt").exists()
