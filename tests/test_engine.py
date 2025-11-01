from uabt.engine import config_from_dict, posterior_probability
import json

def test_posterior_runs():
    cfg = config_from_dict(json.loads(open("configs/uabt_params.json","r",encoding="utf-8").read()))
    p = posterior_probability(cfg)
    assert 0.0 < p < 1.0
