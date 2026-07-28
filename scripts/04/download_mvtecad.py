from pathlib import Path

from anomalib.data import MVTecAD

DATASET_ROOT = Path("./datasets/MVTecAD")
CATEGORY = "bottle"

category_dir = DATASET_ROOT / CATEGORY

if category_dir.exists():
    print("✅ Dataset já disponível.")
else:
    print("⬇️ Baixando dataset...")

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category=CATEGORY,
    )

    datamodule.prepare_data()

    print("✅ Download concluído.")

train_good = list(
    (category_dir / "train" / "good").glob("*.*")
)

test_images = list(
    (category_dir / "test").rglob("*.*")
)

print()
print(f"Treino: {len(train_good)} imagens")
print(f"Teste : {len(test_images)} imagens")