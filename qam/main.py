import typer

from .inference.main import app as infer_app
from .preprocessing.main import app as preprocess_app
from .training.main import app as train_app

app = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(preprocess_app, name="prepare")
app.add_typer(infer_app, name="infer")
