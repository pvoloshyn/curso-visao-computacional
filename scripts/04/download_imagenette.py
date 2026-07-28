from pathlib import Path
import shutil
import tarfile
import urllib.request

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

DATASET_DIR = Path("./datasets")
TARGET_DIR = DATASET_DIR / "imagenette"

ARCHIVE = DATASET_DIR / "imagenette2-320.tgz"
TEMP_DIR = DATASET_DIR / "imagenette2-320"


def already_downloaded() -> bool:
    return (
        TARGET_DIR.exists()
        and any(TARGET_DIR.iterdir())
    )


def download_file():
    print("⬇️ Baixando Imagenette...")
    urllib.request.urlretrieve(URL, ARCHIVE)
    print("✅ Download concluído")


def extract_file():
    print("📦 Extraindo arquivos...")

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATASET_DIR)

    print("✅ Extração concluída")


def reorganize():
    print("🔧 Preparando estrutura esperada pelo EfficientAD...")

    train_dir = TEMP_DIR / "train"

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            destination = TARGET_DIR / class_dir.name

            if destination.exists():
                shutil.rmtree(destination)

            shutil.move(
                str(class_dir),
                str(destination),
            )

    print("✅ Estrutura preparada")


def cleanup():
    print("🧹 Limpando arquivos temporários...")

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    if ARCHIVE.exists():
        ARCHIVE.unlink()

    print("✅ Limpeza concluída")


def main():
    if already_downloaded():
        print("✅ Imagenette já disponível")
        print(TARGET_DIR.resolve())
        return

    DATASET_DIR.mkdir(exist_ok=True)

    download_file()
    extract_file()
    reorganize()
    cleanup()

    print()
    print("=" * 50)
    print("Imagenette instalada com sucesso")
    print("=" * 50)

    classes = sum(
        1
        for item in TARGET_DIR.iterdir()
        if item.is_dir()
    )

    print(f"Classes encontradas: {classes}")
    print(f"Diretório: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
