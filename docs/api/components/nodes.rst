Component Nodes
===============

Data Sources
------------

.. automodule:: nodeml.components.nodes.data_sources.tabular_csv_fetcher
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

.. automodule:: nodeml.components.nodes.models.linear_regression
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.models.random_forest_regressor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.models.random_forest_classifier
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.models.gradient_boosting_regressor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.models.gradient_boosting_classifier
   :members:
   :undoc-members:
   :show-inheritance:

Transforms
----------

Encodings
~~~~~~~~~

.. automodule:: nodeml.components.nodes.transforms.encodings.one_hot_encoding
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.encodings.label_encoding
   :members:
   :undoc-members:
   :show-inheritance:

Feature Selection
~~~~~~~~~~~~~~~~~

.. automodule:: nodeml.components.nodes.transforms.feature_selection.correlation_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.feature_selection.data_category_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.feature_selection.missing_rate_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.feature_selection.variance_filter
   :members:
   :undoc-members:
   :show-inheritance:

Filters
~~~~~~~

.. automodule:: nodeml.components.nodes.transforms.filters.iqr_outlier_filter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.filters.zscore_outlier_filter
   :members:
   :undoc-members:
   :show-inheritance:

Imputations
~~~~~~~~~~~

.. automodule:: nodeml.components.nodes.transforms.imputations.categorical_imputation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.imputations.numerical_imputation
   :members:
   :undoc-members:
   :show-inheritance:

Operations
~~~~~~~~~~

.. automodule:: nodeml.components.nodes.transforms.operations.feature_concatenate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.operations.row_concatenate
   :members:
   :undoc-members:
   :show-inheritance:

Scalers
~~~~~~~

.. automodule:: nodeml.components.nodes.transforms.scalers.standard_scaler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.scalers.min_max_scaler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.transforms.scalers.robust_scaler
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

Classification
~~~~~~~~~~~~~~

.. automodule:: nodeml.components.nodes.metrics.classification.accuracy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.classification.auroc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.classification.f1_score
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.classification.precision
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.classification.recall
   :members:
   :undoc-members:
   :show-inheritance:

Regression
~~~~~~~~~~

.. automodule:: nodeml.components.nodes.metrics.regression.mae
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.regression.mse
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.regression.r2_score
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nodeml.components.nodes.metrics.regression.mape
   :members:
   :undoc-members:
   :show-inheritance:

Sinks
-----

See :mod:`nodeml.core.nodes.data_sink.sink` in the core API reference.
