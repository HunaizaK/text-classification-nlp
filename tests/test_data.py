import pandas as pd
import pytest

from newsclf.config import load_config
from newsclf.data import clean_text, load_csv, make_splits, stratified_take, validate


def test_clean_text_undoes_ag_news_encoding():
    raw = r"A second\team of  #36;10 million, it #39;s &lt;b&gt;big&lt;/b&gt;"
    assert clean_text(raw) == "A second team of $10 million, it's big"


def test_clean_text_restores_named_entities():
    assert clean_text("The EU said it quot;hoped quot; AT amp;T would agree") == 'The EU said it"hoped" AT&T would agree'


def test_load_csv_shifts_labels_and_joins_fields(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text('"1","Title one","Body one"\n"4","Title two","It #39;s two"\n')
    df = load_csv(path)
    assert df["label"].tolist() == [0, 3]
    assert df["text"].tolist() == ["Title one. Body one", "Title two. It's two"]


def test_validate_rejects_bad_labels():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"text": ["x"], "label": [7]}))


def test_stratified_take_keeps_class_balance(toy_df):
    sub = stratified_take(toy_df, 40, seed=0)
    assert len(sub) == 40
    assert sub["label"].value_counts().tolist() == [10, 10, 10, 10]


def test_splits_remove_duplicates_and_test_leakage(toy_df):
    cfg = load_config()
    cfg.data.train_size, cfg.data.val_size, cfg.data.test_size = 40, 20, 20
    test = toy_df.iloc[:30]
    train = pd.concat([toy_df, toy_df.iloc[:5]])  # duplicates + every test article also in train
    splits = make_splits(train, test, cfg)
    assert not set(splits.train_pool["text"]) & set(test["text"])
    assert splits.train_pool["text"].is_unique
    assert len(splits.val) == 20


def test_full_config_overrides_only_listed_keys():
    quick, full = load_config(), load_config(full=True)
    assert full.data.train_size > quick.data.train_size
    assert full.transformer.learning_rate == quick.transformer.learning_rate
    assert full.artifacts_dir != quick.artifacts_dir
