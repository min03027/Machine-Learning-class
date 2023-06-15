from __future__ import annotations
import collections
import csv
import datetime
import enum
from math import isclose
from pathlib import Path
from typing import (
    cast,
    Any,
    Optional,
    Union,
    Iterator,
    Iterable,
    Counter,
    Callable,
    Protocol,
)
import weakref



class Sample:
    """Abstract superclass for all samples."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def __eq__(self, other: Any) -> bool:
        if not issubclass(type(other), Sample):
            return False
        if self.hash() != other.hash():
            return False
        return all(
            [
                self.sepal_length == other.sepal_length,
                self.sepal_width == other.sepal_width,
                self.petal_length == other.petal_length,
                self.petal_width == other.petal_width,
            ]
        )


class KnownSample(Sample):
    """Abstract superclass for testing and training data, the species is set externally."""

    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(f"Invalid purpose: {purpose!r}: {purpose_enum}")
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.species = species
        self.species = species
        self._classification: Optional[str] = None


    def matches(self) -> bool:
        return self.species == self.classification


    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["purpose"] = f"{self.purpose.value}"
        base_attributes["species"] = f"{self.species!r}"
        if self.purpose == Purpose.Testing and self._classification:
            base_attributes["classification"] = f"{self._classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


class UnknownSample(Sample):
    """A sample provided by a User, to be classified."""

    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        classification: Optional[str] = None,
    ) -> None:
        super().__init__(
            species=species,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.classification = classification


    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["classification"] = f"{self.classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"



class Distance:
    """A distance computation"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError


class Chebyshev(Distance):
    """
    Computes the Chebyshev distance between two samples.
    ::
        >>> from math import isclose
        >>> from model import TrainingKnownSample, UnknownSample, Chebyshev
        >>> s1 = TrainingKnownSample(
        ...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
        >>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})
        >>> algorithm = Chebyshev()
        >>> isclose(3.3, algorithm.distance(s1, u))
        True
    """

    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class Minkowski(Distance):
    """An abstraction to provide a way to implement Manhattan and Euclidean."""

    m: int

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            sum(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )


class Euclidean(Minkowski):
    m = 2


class Manhattan(Minkowski):
    m = 1


class Sorensen(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )


class Reduce_Function(Protocol):
    """Define a callable object with specific parameters."""

    def __call__(self, values: list[float]) -> float:
        pass


class Minkowski_2(Distance):
    """A generic way to implement Manhattan, Euclidean, and Chebyshev.

    ::

        >>> from math import isclose
        >>> from model import KnownSample, Purpose, UnknownSample, Minkowski_2


        >>> class CD(Minkowski_2):
        ...     m = 1
        ...     reduction = max

        >>> s1 = KnownSample(
        ...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa",
        ...     purpose=Purpose.Training)
        >>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

        >>> algorithm = CD()
        >>> isclose(3.3, algorithm.distance(s1, u))
        True

    """

    m: int
    reduction: Reduce_Function

    def distance(self, s1: Sample, s2: Sample) -> float:
        # Required to prevent Python from passing `self` as the first argument.
        summarize = self.reduction
        return (
            summarize(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )


class Hyperparameter:
    """A hyperparameter value and the overall quality of the classification."""

    def __init__(self, k: int, algorithm: "Distance", training: "TrainingData") -> None:
        self.k = k
        self.algorithm = algorithm
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Run the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Union[UnknownSample, TestingKnownSample]) -> str:
        """The k-NN algorithm"""
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No TrainingData object")
        distances: list[tuple[float, TrainingKnownSample]] = sorted(
            (self.algorithm.distance(sample, known), known)
            for known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


test_shuffling = """
>>> import random
>>> from pprint import pprint
>>> data = [
...     {
...         "sepal_length": i + 0.1,
...         "sepal_width": i + 0.2,
...         "petal_length": i + 0.3,
...         "petal_width": i + 0.4,
...         "species": f"sample {i}",
...     }
...     for i in range(10)
... ]
>>> random.seed(42)
>>> ssp = ShufflingSamplePartition(data)
>>> pprint(ssp.testing)
[TestingKnownSample(sepal_length=0.1, sepal_width=0.2, petal_length=0.3, petal_width=0.4, species='sample 0', classification=None, ),
 TestingKnownSample(sepal_length=1.1, sepal_width=1.2, petal_length=1.3, petal_width=1.4, species='sample 1', classification=None, )]
"""





# Special case, we don't *often* test abstract superclasses.
# In this example, however, we can create instances of the abstract class.
test_Sample = """
>>> x = Sample(1, 2, 3, 4)
>>> x
Sample(sepal_length=1, sepal_width=2, petal_length=3, petal_width=4, )
>>> y = Sample(1, 2, 3, 4)
>>> x is y
False
>>> x.hash() == y.hash()
True
>>> x == y
True
>>> z = Sample(2, 3, 4, 1)
>>> x.hash() == z.hash()
True
>>> x == z
False
>>> a = Sample(1, 1, 2, 2)
>>> x.hash() == a.hash()
False
"""

test_TrainingKnownSample = """
>>> s1 = TrainingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> s1
TrainingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species='Iris-setosa', )
"""

test_TestingKnownSample = """
>>> s2 = TestingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> s2
TestingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species='Iris-setosa', classification=None, )
>>> s2.classification = "wrong"
>>> s2
TestingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species='Iris-setosa', classification='wrong', )
"""

test_UnknownSample = """
>>> u = UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, )
>>> u
UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, )
"""

test_Chebyshev = """
>>> s1 = TrainingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})
>>> algorithm = Chebyshev()
>>> isclose(3.3, algorithm.distance(s1, u))
True
"""

test_Euclidean = """
>>> s1 = TrainingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})
>>> algorithm = Euclidean()
>>> isclose(4.50111097, algorithm.distance(s1, u))
True
"""

test_Manhattan = """
>>> s1 = TrainingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})
>>> algorithm = Manhattan()
>>> isclose(7.6, algorithm.distance(s1, u))
True
"""

test_Sorensen = """
>>> s1 = TrainingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})
>>> algorithm = Sorensen()
>>> isclose(0.2773722627, algorithm.distance(s1, u))
True
"""

test_Hyperparameter = """
>>> td = TrainingData('test')
>>> s2 = TestingKnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa")
>>> td.testing = [s2]
>>> t1 = TrainingKnownSample(**{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa"})
>>> t2 = TrainingKnownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor"})
>>> td.training = [t1, t2]
>>> h = Hyperparameter(k=3, algorithm=Chebyshev(), training=td)
>>> u = UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
>>> h.classify(u)
'Iris-setosa'
>>> h.test()
>>> print(f"data={td.name!r}, k={h.k}, quality={h.quality}")
data='test', k=3, quality=1.0
"""

test_TrainingData = """
>>> td = TrainingData('test')
>>> raw_data = [
... {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa"},
... {"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor"},
... ]
>>> td.load(raw_data)
>>> h = Hyperparameter(k=3, algorithm=Chebyshev(), training=td)
>>> len(td.training)
1
>>> len(td.testing)
1
>>> td.test(h)
>>> print(f"data={td.name!r}, k={h.k}, quality={h.quality}")
data='test', k=3, quality=0.0
"""

__test__ = {name: case for name, case in globals().items() if name.startswith("test_")}
