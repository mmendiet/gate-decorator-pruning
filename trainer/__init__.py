from trainer.normal import NormalTrainer, MutualTrainer, MutualTrainerBN
from config import cfg

def get_trainer():
    pair = {
        'normal': NormalTrainer,
        'mutual': MutualTrainer,
        'mutualBN': MutualTrainerBN
    }
    assert (cfg.train.trainer in pair)

    return pair[cfg.train.trainer]()
