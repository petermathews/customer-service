"""Image classification.

When a customer attaches a photo (a damaged item, a screenshot of an error),
the system has to work from pixels rather than text. That is image
classification, the fourth problem type in this comparison.

Training on real product photos requires a labelled image set that cannot be
published, so this uses a self contained stand in: scikit-learn's `digits`
dataset of 1,797 genuine 8x8 handwritten digit images, trained with an MLP. The
pipeline is the one used on product photos (load pixels, split, train a neural
network, evaluate); only the images differ. In the agent, this classifier
stands in for inspecting the attached photo and deciding what it shows.
"""

from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def train_image_classifier():
    """Returns a fitted classifier that maps pixels to a label, plus its test accuracy."""
    digits = load_digits()
    X, y = digits.images.reshape(len(digits.images), -1), digits.target  # flatten 8x8 to 64 features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42),
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


if __name__ == "__main__":
    _, acc = train_image_classifier()
    print(f"Image classifier (digits stand in) test accuracy: {acc:.1%}")
