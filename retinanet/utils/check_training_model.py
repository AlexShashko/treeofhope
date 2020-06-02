def check_training_model(model):
    """ Assert that the model is a training model.
    """
    assert(all(output in model.output_names for output in ['regression', 'classification'])), \
        f"Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {model.output_names})."
