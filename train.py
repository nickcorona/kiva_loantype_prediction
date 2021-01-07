import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder  # sometimes needed
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess
from helpers import encode_dates

from helpers import loguniform

df = pd.read_csv(
    r"data\kiva_loans.csv",
    parse_dates=[
        "posted_time",
        "disbursed_time",
        "funded_time",
        "date",
    ],
    index_col="id",
)
df.info()

y = df["repayment_interval"].replace(
    ["irregular", "bullet", "monthly", "weekly"], [3, 0, 2, 1]
)
X = df.drop(["repayment_interval", "use", "tags"], axis=1)

obj_cols = X.select_dtypes("object").columns
X[obj_cols] = X[obj_cols].astype("category")

print((X[obj_cols].nunique() / len(df)).sort_values())


def get_male_count(x):
    count = 0
    for gender in x.split(", "):
        if gender == "male":
            count += 1
    return count


def get_female_count(x):
    count = 0
    for gender in x.split(", "):
        if gender == "female":
            count += 1
    return count


X["male_count"] = X["borrower_genders"].apply(get_male_count)
X["female_count"] = X["borrower_genders"].apply(get_female_count)

X = encode_dates(X, "posted_time")
X = encode_dates(X, "disbursed_time")
X = encode_dates(X, "funded_time")
X = encode_dates(X, "date")

SEED = 0
SAMPLE_SIZE = 10000

Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = lgb.Dataset(Xt, yt, free_raw_data=False)
np.random.seed(SEED)
sample_idx = np.random.choice(Xt.index, size=SAMPLE_SIZE)
Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
ds = lgb.Dataset(Xs, ys)
dv = lgb.Dataset(Xv, yv, free_raw_data=False)


OBJECTIVE = "multiclass"
METRIC = "multi_logloss"
MAXIMIZE = False
EARLY_STOPPING_ROUNDS = 10
MAX_ROUNDS = 10000
REPORT_ROUNDS = 100

params = {
    "objective": OBJECTIVE,
    "metric": METRIC,
    "verbose": -1,
    "num_classes": 4,
    "n_jobs": 6,
}

model = lgb.train(
    params,
    ds,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

best_etas = {"learning_rate": [], "score": []}

for _ in range(30):
    eta = loguniform(-2, 0)
    best_etas["learning_rate"].append(eta)
    params["learning_rate"] = eta
    model = lgb.train(
        params,
        ds,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_etas["score"].append(model.best_score["valid"][METRIC])

best_eta_df = pd.DataFrame.from_dict(best_etas)
lowess_data = lowess(
    best_eta_df["score"],
    best_eta_df["learning_rate"],
)

rounded_data = lowess_data.copy()
rounded_data[:, 1] = rounded_data[:, 1].round(4)
rounded_data = rounded_data[::-1]  # reverse to find first best
# maximize or minimize metric
if MAXIMIZE:
    best = np.argmax
else:
    best = np.argmin
best_eta = rounded_data[best(rounded_data[:, 1]), 0]

# plot relationship between learning rate and performance, with an eta selected just before diminishing returns
# use log scale as it's easier to observe the whole graph
sns.lineplot(x=lowess_data[:, 0], y=lowess_data[:, 1])
plt.xscale("log")
print(f"Good learning rate: {best_eta:4f}")
plt.axvline(best_eta, color="orange")
plt.title("Smoothed relationship between learning rate and metric.")
plt.xlabel("learning rate")
plt.ylabel(METRIC)
plt.show()

params["learning_rate"] = 0.02

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

threshold = 0.75
corr = Xt.corr(method="kendall")
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
upper = upper.stack()
high_upper = upper[(abs(upper) > threshold)]
abs_high_upper = abs(high_upper).sort_values(ascending=False)
pairs = abs_high_upper.index.to_list()
print(f"Correlated features: {pairs if len(pairs) > 0 else None}")

# drop correlated features
best_score = model.best_score["valid"][METRIC]
print(f"starting score: {best_score:.4f}")
drop_dict = {pair: [] for pair in pairs}
correlated_features = set()
for pair in pairs:
    for feature in pair:
        correlated_features.add(feature)
        Xt, Xv, yt, yv = train_test_split(
            X.drop(correlated_features, axis=1), y, random_state=SEED
        )
        dt = lgb.Dataset(Xt, yt, silent=True)
        dv = lgb.Dataset(Xv, yv, silent=True)
        drop_model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        drop_dict[pair].append(drop_model.best_score["valid"][METRIC])
        correlated_features.remove(feature)  # remove from drop list
    pair_min = np.min(drop_dict[pair])
    if pair_min <= best_score:
        drop_feature = pair[
            np.argmin(drop_dict[pair])
        ]  # add to drop_feature the one that reduces score
        best_score = pair_min
        correlated_features.add(drop_feature)
print(f"ending score: {best_score:.4f}")
print(
    f"dropped features: {correlated_features if len(correlated_features) > 0 else None}"
)

X = X.drop(correlated_features, axis=1)
Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = lgb.Dataset(Xt, yt, silent=True)
dv = lgb.Dataset(Xv, yv, silent=True)

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

sorted_features = [
    feature
    for _, feature in sorted(
        zip(model.feature_importance(importance_type="gain"), dt.feature_name),
        reverse=False,
    )
]

best_score = model.best_score["valid"][METRIC]
print(f"starting score: {best_score:.4f}")
unimportant_features = []
for feature in sorted_features:
    unimportant_features.append(feature)
    Xt, Xv, yt, yv = train_test_split(
        X.drop(unimportant_features, axis=1), y, random_state=SEED
    )
    dt = lgb.Dataset(Xt, yt, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)

    drop_model = lgb.train(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    score = drop_model.best_score["valid"][METRIC]
    if score > best_score:
        del unimportant_features[-1]  # remove from drop list
        print(f"Dropping {feature} worsened score to {score:.4f}.")
        break
    else:
        best_score = score
print(f"ending score: {best_score:.4f}")
print(
    f"dropped features: {unimportant_features if len(unimportant_features) > 0 else None}"
)

X = X.drop(unimportant_features, axis=1)
Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = lgb.Dataset(Xt, yt, silent=True)
dv = lgb.Dataset(Xv, yv, silent=True)

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

import optuna.integration.lightgbm as lgb

dt = lgb.Dataset(Xt, yt, silent=True)
dv = lgb.Dataset(Xv, yv, silent=True)

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    verbose_eval=False,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

score = model.best_score["valid"][METRIC]

best_params = model.params
print("Best params:", best_params)
print(f"  {METRIC} = {score}")
print("  Params: ")
for key, value in best_params.items():
    print(f"    {key}: {value}")

import lightgbm as lgb

model = lgb.train(
    best_params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

lgb.plot_importance(model, importance_type="gain", grid=False)
plt.show()
