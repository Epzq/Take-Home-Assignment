import os
import re
import tempfile
import pytest
import pandas as pd
import numpy as np
from src.utils import load_and_version_data
from src.utils import check_model_promotion, evaluate_model, set_seed


def test_model_promotion_logic():
    """Test evaluating threshold gates for deployment."""
    # Pass
    assert (check_model_promotion({"accuracy": 0.85, "f1_score": 0.80}, min_accuracy=0.70, min_f1=0.8) is True )
    # Fails accuracy
    assert (check_model_promotion({"accuracy": 0.65, "f1_score": 0.80}, min_accuracy=0.70 ) is False)
    # Fails F1
    assert (check_model_promotion({"accuracy": 0.75, "f1_score": 0.60}, min_accuracy=0.70, min_f1=0.68,) is False )

def test_data_versioning_determinism_and_uniqueness():
    """Unit test to verify timestamped hash generation, directory reuse, and uniqueness."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_csv_dir = os.path.join(tmp_dir, "raw_csv")
        save_dir = os.path.join(tmp_dir, "data")
        os.makedirs(raw_csv_dir, exist_ok=True)
        # Create mock dataset
        train_df_a = pd.DataFrame({
            "Class Index": [1, 2, 3],
            "Title": ["Title 1", "Title 2", "Title 3"],
            "Description": ["Desc 1", "Desc 2", "Desc 3"],
        })
        test_df = pd.DataFrame({
            "Class Index": [4],
            "Title": ["Test Title"],
            "Description": ["Test Desc"],
        })
        train_df_a.to_csv(os.path.join(raw_csv_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_csv_dir, "test.csv"), index=False)

        _, _, version_tag_a1 = load_and_version_data(
            raw_csv_dir=raw_csv_dir, save_dir=save_dir, train_samples=3, test_samples=1
        )
        # Verify format matches ag_news_{YYYYMMDD}_{HHMMSS}_v{HASH}
        pattern = r"^ag_news_\d{8}_\d{6}_v[a-f0-9]{10}$"
        assert re.match(pattern, version_tag_a1), f"Version tag {version_tag_a1} error."

        # Check directory exists on disk
        expected_dir_a1 = os.path.join(save_dir, version_tag_a1)
        assert os.path.exists(expected_dir_a1)
        assert os.path.exists(os.path.join(expected_dir_a1, "train"))
        assert os.path.exists(os.path.join(expected_dir_a1, "test"))

        # second run for same data, check hash
        hash_a1 = version_tag_a1.split("_v")[-1]
        _, _, version_tag_a2 = load_and_version_data(
            raw_csv_dir=raw_csv_dir, save_dir=save_dir, train_samples=3, test_samples=1
        )
        assert version_tag_a1 == version_tag_a2

        # Change Data to check for hash change
        train_df_b = pd.DataFrame({
            "Class Index": [1, 2, 3],
            "Title": ["Title 1", "Title 2", "Title change"],
            "Description": ["Desc 1", "Desc 2", "Desc 3"],
        })
        train_df_b.to_csv(os.path.join(raw_csv_dir, "train.csv"), index=False)
        _, _, version_tag_b = load_and_version_data(
            raw_csv_dir=raw_csv_dir, save_dir=save_dir, train_samples=3, test_samples=1
        )

        hash_b = version_tag_b.split("_v")[-1]

        # Assert modified data generates a new folder with a distinct hash
        assert hash_a1 != hash_b
        assert version_tag_a1 != version_tag_b