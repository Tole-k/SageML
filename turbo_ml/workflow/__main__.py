from typing import Tuple
from prefect import flow
from turbo_ml.workflow import (generate_training_parameters, train_meta_model,
                               save_meta_model, load_algorithms_evaluations, test_TurboML)
from turbo_ml.meta_learning.dataset_parameters.sota import get_sota_meta_features
from turbo_ml.utils import options


@flow(name='Full Meta Model Workflow', log_prints=True)
def full_pipeline() -> Tuple[int]:
    evaluations = load_algorithms_evaluations('algorithm_results.csv')
    training_parameters = generate_training_parameters(output_path=None, meta_data_extractor=get_sota_meta_features(options.meta_features))
    model, preprocessor = train_meta_model(
        training_parameters, evaluations, 2500)
    save_meta_model(model, preprocessor, 'new_model')
    test_TurboML('new_model/', get_sota_meta_features(options.meta_features))
    return model, preprocessor


if __name__ == '__main__':
    full_pipeline()
