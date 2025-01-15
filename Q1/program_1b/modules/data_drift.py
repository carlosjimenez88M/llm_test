'''

'''


#=====================#
# ---- libraries ---- #
#=====================#

from evidently import ColumnMapping
from evidently.metrics import DataDriftTable
from evidently.report import Report

#=========================#
# ---- main function ---- #
#=========================#

def detect_data_drift(reference_data, current_data):
    """
    Detects and reports data drift between a reference dataset and a current dataset.

    This function uses the Evidently library to generate a report that identifies
    potential data drift issues in the provided datasets. The report is configured
    to compare features, evaluate drift statistics, and present the results in an
    interactive format.

    Args:
        reference_data (pd.DataFrame):
            A Pandas DataFrame representing the reference dataset. This dataset serves
            as the baseline for comparison.

        current_data (pd.DataFrame):
            A Pandas DataFrame representing the current dataset. This dataset is
            evaluated against the reference dataset to detect potential drift.

    Returns:
        None: Displays an inline HTML report with the results of the data drift analysis.

    Raises:
        ValueError: If the provided datasets are not valid Pandas DataFrames or
            contain incompatible structures.

    Example:
       # >>> from pandas import DataFrame
       # >>> reference = DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
       # >>> current = DataFrame({"feature1": [1, 2, 2], "feature2": [4, 5, 7]})
       # >>> detect_data_drift(reference, current)
    """
    column_mapping = ColumnMapping()
    report = Report(metrics=[DataDriftTable()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    return report.show(mode="inline")