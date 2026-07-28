import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time

from anomalib.data import MVTecAD
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize 
from anomalib.engine import Engine

from anomalib.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger


def main():
    # ==========================================================
    # Configurações
    # ==========================================================

    DATASET_ROOT = Path("./datasets/MVTecAD")
    CATEGORY = "bottle"

    CHECKPOINT_DIR = Path("./checkpoints")
    LOG_DIR = Path("./logs")

    MAX_EPOCHS = 20


    # ==========================================================
    # Verificação do dataset
    # ==========================================================

    bottle_dir = DATASET_ROOT / CATEGORY

    if not bottle_dir.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {bottle_dir}"
        )

    train_good = list(
        (bottle_dir / "train" / "good").glob("*.*")
    )

    test_images = list(
        (bottle_dir / "test").rglob("*.*")
    )

    print("=" * 60)
    print("MVTecAD")
    print("=" * 60)

    print(f"Categoria: {CATEGORY}")
    print(f"Treino (good): {len(train_good)}")
    print(f"Teste total : {len(test_images)}")
    print()


    # ==========================================================
    # DataModule
    # ==========================================================

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category=CATEGORY,
        train_batch_size=1,
        eval_batch_size=32,
        num_workers=0,
    )


    # ==========================================================
    # Modelo
    # ==========================================================

    model = EfficientAd(
        model_size=EfficientAdModelSize.S
    )


    # ==========================================================
    # Callbacks
    # ==========================================================

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="efficientad-bottle-{epoch:02d}",
        every_n_epochs=1,
        save_top_k=1,
    )

    progress_bar = TQDMProgressBar(
        refresh_rate=1
    )

    logger = CSVLogger(
        save_dir=str(LOG_DIR),
        name="efficientad_bottle"
    )


    # ==========================================================
    # Engine
    # ==========================================================

    engine = Engine(
        default_root_dir='modelos',
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            progress_bar,
        ],
    )


    # ==========================================================
    # Treinamento
    # ==========================================================

    print()
    print("Iniciando treinamento...")
    print()

    start = time.time()

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    elapsed = (time.time() - start) / 60

    print()
    print("=" * 60)
    print("TREINAMENTO FINALIZADO")
    print("=" * 60)

    print(f"Tempo total: {elapsed:.2f} minutos")
    print(f"Melhor checkpoint: {checkpoint_callback.best_model_path}")
    print()

if __name__ == '__main__':
    main()