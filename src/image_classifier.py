"""Image classification mini-lesson.

When a customer attaches a *photo* (a damaged item, a screenshot of an error),
the system needs to look at pixels, not text. That's image classification — one
of the four ML concepts a coach has to be fluent in.

Training on real product photos needs a labelled image set we can't ship here,
so this is a self-contained stand-in: scikit-learn's `digits` dataset (1,797
real 8×8 handwritten-digit images) trained with an MLP. The *pipeline* is
exactly what you'd use on product photos — load pixels → split → train a neural
net → evaluate — only the images differ. In the agent, this classifier stands
in for "look at the attached photo and decide what it shows."
"""

from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def train_image_classifier():
    """Returns a fitted pixels-in → label-out classifier and its test accuracy."""
    digits = load_digits()
    X, y = digits.images.reshape(len(digits.images), -1), digits.target  # flatten 8x8 → 64
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
    print(f"Image classifier (digits stand-in) test accuracy: {acc:.1%}")
