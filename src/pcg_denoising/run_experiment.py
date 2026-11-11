import argparse
from pathlib import Path

import pcg_denoising.config as config
from pcg_denoising.utils.progress import Progress

import pcg_denoising.splits as SP
import pcg_denoising.build_datasets as BD
import pcg_denoising.network_training as NT
import pcg_denoising.denoise_pcg as DP
import pcg_denoising.report as RP

import pcg_denoising.utils.noise as NS
import pcg_denoising.utils.evaluation as EV
from pcg_denoising.utils.utils import transform_args


def make_noisy_audio(cfg, fold, d, train_noise):
    NS.add_noise(fold.train + fold.val, cfg.directory.pcg_raw, train_noise, d.train_noisy,
                 dB=cfg.noise.training_noise, sr=cfg.audio.sr, noise_type="Sum",
                 seed=cfg.seed.noise_training_seed + fold.index)


def build_datasets(cfg, fold, d):
    raw = cfg.directory.pcg_raw
    name = cfg.transform.name
    noisy_train = [Path(d.train_noisy) / f for f in fold.train]
    noisy_val = [Path(d.train_noisy) / f for f in fold.val]

    if name == "STFT":
        BD.build_spectrograms(fold.paths("train", raw), noisy_train, d.train_h5, *transform_args(cfg))
        BD.build_spectrograms(fold.paths("val", raw), noisy_val, d.val_h5, *transform_args(cfg))
    elif name in ("WSST", "CWT"):
        BD.build_wavelet(name, fold.paths("train", raw), noisy_train, d.train_h5, cfg.transform.active.overlap, *transform_args(cfg))
        BD.build_wavelet(name, fold.paths("val", raw), noisy_val, d.val_h5, cfg.transform.active.overlap, *transform_args(cfg))
    elif name == "S":
        BD.build_stockwell(fold.paths("train", raw), noisy_train, d.train_h5, *transform_args(cfg))
        BD.build_stockwell(fold.paths("val", raw), noisy_val, d.val_h5, *transform_args(cfg))


def train_fold(cfg, d):
    tr = cfg.training
    NT.unetTraining(d.root, d.model, d.model_history, tr.learning_rate, tr.batch_size, tr.epochs, cfg.seed.network_initialization_seed)


def evaluate_fold(cfg, fold, d, test_noise, csv):
    model = NT.load_model(d.model)
    targets = fold.paths("test", cfg.directory.pcg_raw)
    done = EV.completed(csv)
    for noise_type in cfg.noise.evaluation_noise_type:
        for level in cfg.noise.evaluation_noise:
            if (fold.index, noise_type, level) in done:
                continue
            noisy, kinds = NS.add_noise(fold.test, cfg.directory.pcg_raw, test_noise, d.test_noisy(noise_type, level), level, cfg.audio.sr, noise_type, seed=cfg.seed.noise_test_seed + fold.index)
            preds, avg_time = DP.runNetwork(cfg.transform.name, model, fold.test, d.test_noisy(noise_type, level), d.test_prediction(noise_type, level), *transform_args(cfg))
            rows = EV.score_condition(fold.index, noise_type, level, fold.test, targets, noisy, preds, kinds, cfg.audio.sr, avg_time)
            EV.append_results(csv, rows)


def run_experiment(cfg, mode):
    run_dir, csv, xlsx, sr = cfg.run_dir, cfg.csv_results, cfg.xlsx_results, cfg.audio.sr
    noise_dir = cfg.directory.noise_processed
    progress = Progress(run_dir, mode=mode)

    if progress.restarted:
        csv.unlink(missing_ok=True)

    train_noise = {"PhysioNet": noise_dir / f"PhysioNet_train.wav", 
                   "Speech": noise_dir / f"Speech_train.wav"}

    test_noise = {"PhysioNet": noise_dir / f"PhysioNet_test.wav", 
                  "Speech": noise_dir / f"Speech_test.wav"}

    progress.run("noise_split", NS.split_noise, cfg.directory.physionet_raw, cfg.directory.speech_raw,
                 noise_dir, eval_percentage=cfg.training.dataset_split, sr=sr, seed=cfg.seed.noise_segmentation_seed,
                 deps=(sr, cfg.training.dataset_split, cfg.seed.noise_segmentation_seed),
                 requires=[*train_noise.values(), *test_noise.values()])

    folds_file = cfg.folds_json
    if folds_file.exists() and not progress.restarted:
        folds = SP.load_folds(folds_file, cfg.directory.pcg_raw)
    else:
        folds = SP.create_folds(cfg.directory.pcg_raw, folds_file, strategy="stratified_kfold",
                                k=cfg.training.folds, val_fraction=cfg.training.training_split, seed=cfg.seed.folds_seed)

    noise_deps = (sr, cfg.seed, cfg.noise.training_noise)
    data_deps = (*noise_deps, cfg.audio, cfg.transform)
    train_deps = (*data_deps, cfg.training)

    for fold in folds:
        d, key = cfg.fold_dirs(fold.index), f"fold_{fold.index}"
        train_key, ds_key, noise_key = f"{key}/train", f"{key}/datasets", f"{key}/noise"
        if not progress.done(train_key, deps=train_deps, requires=[d.model]):
            if not progress.done(ds_key, requires=[d.train_h5, d.val_h5]):
                # Rebuild datasets if h5 files are missing
                progress.run(noise_key, make_noisy_audio, cfg, fold, d, train_noise, deps=noise_deps, requires=[d.train_noisy])
                progress.run(ds_key, build_datasets, cfg, fold, d, deps=data_deps, requires=[d.train_h5, d.val_h5])
            progress.run(train_key, train_fold, cfg, d, deps=train_deps, requires=[d.model])

        if not cfg.training.keep_datasets:
            # If True, keep h5 files
            progress.release(ds_key, [d.train_h5, d.val_h5])
            progress.release(noise_key, [d.train_noisy])

        evaluate_fold(cfg, fold, d, test_noise, csv)

        RP.write_report(csv, xlsx, settings={"Transform": cfg.transform.name, "Learning Rate": cfg.training.learning_rate, 
                                             "Batch Size": cfg.training.batch_size, "Epochs": cfg.training.epochs, 
                                             "Train SNR (dB)": cfg.noise.training_noise, "Network": "U-Net"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None, help="override tomls with a new configuration")
    parser.add_argument("--mode", choices=["resume", "restart"], default="resume")
    args = parser.parse_args()

    cfg = config.load_config(override_path=args.experiment)
    run_experiment(cfg, mode=args.mode)


if __name__ == "__main__":
    main()
