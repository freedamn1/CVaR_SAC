#!/usr/bin/env python

import joblib
import os
import os.path as osp
from typing import Any, Callable, Optional, Tuple

def _resolve_itr(fpath: str, itr: str) -> str:
    if itr != "last":
        return str(int(itr))

    candidates = []
    for name in os.listdir(fpath):
        if not name.startswith("simple_save"):
            continue
        suffix = name[len("simple_save") :]
        if suffix == "":
            candidates.append("")
            continue
        try:
            candidates.append(str(int(suffix)))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(f"在目录中未找到 simple_save*：{fpath}")

    numeric = [c for c in candidates if c != ""]
    if numeric:
        return str(max(int(x) for x in numeric))
    return ""


def load_policy(
    fpath: str,
    itr: str = "last",
    deterministic: bool = False,
) -> Tuple[Optional[Any], Callable[[Any], Any], Any]:
    import tensorflow as tf

    from wc_sac.utils.logx import restore_tf_graph

    if tf.__version__.startswith("2"):
        tf.compat.v1.disable_eager_execution()
        tf_sess = tf.compat.v1.Session
        tf_graph = tf.Graph
    else:
        tf_sess = tf.Session
        tf_graph = tf.Graph

    # handle which epoch to load from
    itr_resolved = _resolve_itr(fpath=fpath, itr=str(itr))

    # load the things!
    sess = tf_sess(graph=tf_graph())
    model = restore_tf_graph(sess, osp.join(fpath, "simple_save" + itr_resolved))

    # get the correct op for executing actions
    if deterministic and "mu" in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print("Using deterministic action op.")
        action_op = model["mu"]
    else:
        print("Using default action op.")
        action_op = model["pi"]

    # make function for producing an action given a single state
    def get_action(x):
        x = x.reshape(-1).astype("float32")
        return sess.run(action_op, feed_dict={model["x"]: x[None, :]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, "vars" + itr_resolved + ".pkl"))
        env = state["env"]
    except:
        env = None

    return env, get_action, sess
