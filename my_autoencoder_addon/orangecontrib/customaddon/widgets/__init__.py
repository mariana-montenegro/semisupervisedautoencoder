from .preprocess_widget import OWPreprocessing
from .autoencoder_widget import OWAutoencoder
from .classifier_widget import OWClassifier

WIDGET_HELP_PATH = (
    ('{NAME}', 'index.html'),
)

__all__ = [
    "OWPreprocessing",
    "OWAutoencoder",
    "OWClassifier"
]
