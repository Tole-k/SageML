from sageml import SageML
from datasets import get_iris
from sageml.utils import options


def test_happypath():
    dataset, target = get_iris()
    dataset['target'] = target
    random = dataset.sample(n=6)
    dataset.drop(random.index, inplace=True)
    test = random['target']
    random.drop('target', axis=1, inplace=True)
    sage_ml = SageML(dataset=dataset, target='target',
                       device=options.device, threads=options.threads, hpo_trials=10)
    result = sage_ml(random)
    assert result is not None
    assert len(result) == len(test)
    assert all(i in target for i in result)
    assert sage_ml.model.__class__.__name__ != 'RandomGuesser'


if __name__ == '__main__':
    test_happypath()
